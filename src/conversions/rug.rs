#![cfg(feature = "rug")]
//!  Conversions to and from [rug](https://docs.rs/rug)’s [`Integer`] and [`BigUint`] types.
//!
//! This is useful for converting Python integers when they may not fit in Rust's built-in integer types.
//!
//! # Setup
//!
//! To use this feature, add this to your **`Cargo.toml`**:
//!
//! ```toml
//! [dependencies]
//! rug = "*"
#![doc = concat!("pyo3 = { version = \"", env!("CARGO_PKG_VERSION"),  "\", features = [\"rug\"] }")]
//! ```
//!
//! Note that you must use compatible versions of rug and PyO3.
//! The required rug version may vary based on the version of PyO3.
//!
//! ## Examples
//!
//! Using [`Integer`] to correctly increment an arbitrary precision integer.
//! This is not possible with Rust's native integers if the Python integer is too large,
//! in which case it will fail its conversion and raise `OverflowError`.
//! ```rust
//! use rug::Integer;
//! use pyo3::prelude::*;
//!
//! #[pyfunction]
//! fn add_one(n: Integer) -> Integer {
//!     n + 1
//! }
//!
//! #[pymodule]
//! fn my_module(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
//!     m.add_function(wrap_pyfunction!(add_one, m)?)?;
//!     Ok(())
//! }
//! ```
//!
//! Python code:
//! ```python
//! from my_module import add_one
//!
//! n = 1 << 1337
//! value = add_one(n)
//!
//! assert n + 1 == value
//! ```

use crate::{
    ffi, types::*, FromPyObject, IntoPy, Py, PyAny, PyObject, PyResult, Python, ToPyObject,
};

use rug::{integer::Order, Integer};

// #[cfg(not(Py_LIMITED_API))]
// TODO

#[cfg_attr(docsrs, doc(cfg(feature = "rug")))]
impl ToPyObject for Integer {
    #[cfg(not(Py_LIMITED_API))]
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let bytes: Vec<u8> = self.to_digits(Order::Lsf);
        unsafe {
            let obj = ffi::_PyLong_FromByteArray(bytes.as_ptr().cast(), bytes.len(), 1, 1);
            PyObject::from_owned_ptr(py, obj)
        }
    }

    #[cfg(Py_LIMITED_API)]
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let bytes = Integer::to_signed_bytes_le(self);
        let bytes_obj = PyBytes::new(py, &bytes);
        let kwargs = if 1 > 0 {
            let kwargs = PyDict::new(py);
            kwargs.set_item(crate::intern!(py, "signed"), true).unwrap();
            Some(kwargs)
        } else {
            None
        };
        py.get_type::<PyLong>()
            .call_method("from_bytes", (bytes_obj, "little"), kwargs)
            .expect("int.from_bytes() failed during to_object()") // FIXME: #1813 or similar
            .into()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "rug")))]
impl IntoPy<PyObject> for Integer {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.to_object(py)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "rug")))]
impl<'source> FromPyObject<'source> for Integer {
    fn extract(ob: &'source PyAny) -> PyResult<Integer> {
        let py = ob.py();
        // fast path - checking for subclass of `int` just checks a bit in the type object
        let num_owned: Py<PyLong>;
        let num = if let Ok(long) = ob.downcast::<PyLong>() {
            long
        } else {
            num_owned = unsafe { Py::from_owned_ptr_or_err(py, ffi::PyNumber_Index(ob.as_ptr()))? };
            num_owned.as_ref(py)
        };
        let n_bits = int_n_bits(num)?;
        if n_bits == 0 {
            return Ok(Integer::from(0isize));
        }
        #[cfg(not(Py_LIMITED_API))]
        {
            let mut buffer = int_to_u32_vec(num, (n_bits + 32) / 32, true)?;
            if buffer.last().copied().map_or(false, |last| last >> 31 != 0) {
                // Integer::new takes an unsigned array, so need to convert from two's complement
                // flip all bits, 'subtract' 1 (by adding one to the unsigned array)
                let mut elements = buffer.iter_mut();
                for element in elements.by_ref() {
                    *element = (!*element).wrapping_add(1);
                    if *element != 0 {
                        // if the element didn't wrap over, no need to keep adding further ...
                        break;
                    }
                }
                // ... so just two's complement the rest
                for element in elements {
                    *element = !*element;
                }
                Ok(-Integer::from_digits(&buffer, Order::Lsf))
            } else {
                Ok(Integer::from_digits(&buffer, Order::Lsf))
            }
        }
        #[cfg(Py_LIMITED_API)]
        {
            let bytes = int_to_py_bytes(num, (n_bits + 8) / 8, true)?;
            Ok(Integer::from_signed_bytes_le(bytes.as_bytes()))
        }
    }
}

#[cfg(not(Py_LIMITED_API))]
#[inline]
fn int_to_u32_vec(long: &PyLong, n_digits: usize, is_signed: bool) -> PyResult<Vec<u32>> {
    let mut buffer = Vec::with_capacity(n_digits);
    unsafe {
        crate::err::error_on_minusone(
            long.py(),
            ffi::_PyLong_AsByteArray(
                long.as_ptr().cast(),
                buffer.as_mut_ptr() as *mut u8,
                n_digits * 4,
                1,
                is_signed.into(),
            ),
        )?;
        buffer.set_len(n_digits)
    };
    buffer
        .iter_mut()
        .for_each(|chunk| *chunk = u32::from_le(*chunk));

    Ok(buffer)
}

#[cfg(Py_LIMITED_API)]
fn int_to_py_bytes(long: &PyLong, n_bytes: usize, is_signed: bool) -> PyResult<&PyBytes> {
    use crate::intern;
    let py = long.py();
    let kwargs = if is_signed {
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "signed"), true)?;
        Some(kwargs)
    } else {
        None
    };
    let bytes = long.call_method(
        intern!(py, "to_bytes"),
        (n_bytes, intern!(py, "little")),
        kwargs,
    )?;
    Ok(bytes.downcast()?)
}

#[inline]
fn int_n_bits(long: &PyLong) -> PyResult<usize> {
    let py = long.py();
    #[cfg(not(Py_LIMITED_API))]
    {
        // fast path
        let n_bits = unsafe { ffi::_PyLong_NumBits(long.as_ptr()) };
        if n_bits == (-1isize as usize) {
            return Err(crate::PyErr::fetch(py));
        }
        Ok(n_bits)
    }

    #[cfg(Py_LIMITED_API)]
    {
        // slow path
        long.call_method0(crate::intern!(py, "bit_length"))
            .and_then(PyAny::extract)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PyDict, PyModule};
    use indoc::indoc;

    fn rust_fib<T>() -> impl Iterator<Item = T>
    where
        T: From<u16>,
        T: Default,
        for<'a> T: std::ops::Add<&'a T, Output = T>,
        T: Clone,
    {
        let mut f0: T = T::from(1);
        let mut f1: T = T::from(1);
        std::iter::from_fn(move || {
            let f2 = f0.clone() + &f1;
            Some(std::mem::replace(&mut f0, std::mem::replace(&mut f1, f2)))
        })
    }

    fn python_fib(py: Python<'_>) -> impl Iterator<Item = PyObject> + '_ {
        let mut f0 = 1.to_object(py);
        let mut f1 = 1.to_object(py);
        std::iter::from_fn(move || {
            let f2 = f0.call_method1(py, "__add__", (f1.as_ref(py),)).unwrap();
            Some(std::mem::replace(&mut f0, std::mem::replace(&mut f1, f2)))
        })
    }

    #[test]
    fn convert_bigint() {
        Python::with_gil(|py| {
            // check the first 2000 numbers in the fibonacci sequence
            for (py_result, rs_result) in python_fib(py).zip(rust_fib::<Integer>()).take(2000) {
                // Python -> Rust
                assert_eq!(py_result.extract::<Integer>(py).unwrap(), rs_result);
                // Rust -> Python
                assert!(py_result.as_ref(py).eq(&rs_result).unwrap());

                // negate

                let rs_result = rs_result * -1;
                let py_result = py_result.call_method0(py, "__neg__").unwrap();

                // Python -> Rust
                assert_eq!(py_result.extract::<Integer>(py).unwrap(), rs_result);
                // Rust -> Python
                assert!(py_result.as_ref(py).eq(rs_result).unwrap());
            }
        });
    }

    fn python_index_class(py: Python<'_>) -> &PyModule {
        let index_code = indoc!(
            r#"
                class C:
                    def __init__(self, x):
                        self.x = x
                    def __index__(self):
                        return self.x
                "#
        );
        PyModule::from_code(py, index_code, "index.py", "index").unwrap()
    }

    #[test]
    fn convert_index_class() {
        Python::with_gil(|py| {
            let index = python_index_class(py);
            let locals = PyDict::new(py);
            locals.set_item("index", index).unwrap();
            let ob = py.eval("index.C(10)", None, Some(locals)).unwrap();
            let _: Integer = FromPyObject::extract(ob).unwrap();
        });
    }

    #[test]
    fn handle_zero() {
        Python::with_gil(|py| {
            let zero: Integer = 0.to_object(py).extract(py).unwrap();
            assert_eq!(zero, Integer::from(0));
        })
    }

    /// `OverflowError` on converting Python int to Integer, see issue #629
    #[test]
    fn check_overflow() {
        Python::with_gil(|py| {
            macro_rules! test {
                ($T:ty, $value:expr, $py:expr) => {
                    let value = $value;
                    println!("{}: {}", stringify!($T), value);
                    let python_value = value.clone().into_py(py);
                    let roundtrip_value = python_value.extract::<$T>(py).unwrap();
                    assert_eq!(value, roundtrip_value);
                };
            }

            for i in 0..=256usize {
                // test a lot of values to help catch other bugs too
                test!(Integer, Integer::from(i), py);
                test!(Integer, -Integer::from(i), py);
                test!(Integer, Integer::from(1) << i, py);
                test!(Integer, -Integer::from(1) << i, py);
                test!(Integer, (Integer::from(1) << i) + 1u32, py);
                test!(Integer, (-Integer::from(1) << i) + 1u32, py);
                test!(Integer, (Integer::from(1) << i) - 1u32, py);
                test!(Integer, (-Integer::from(1) << i) - 1u32, py);
            }
        });
    }
}
