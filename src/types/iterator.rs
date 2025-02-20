use crate::ffi_ptr_ext::FfiPtrExt;
use crate::py_result_ext::PyResultExt;
use crate::{
    ffi, AsPyPointer, Py2, PyAny, PyDowncastError, PyErr, PyNativeType, PyResult, PyTypeCheck,
};

/// A Python iterator object.
///
/// # Examples
///
/// ```rust
/// use pyo3::prelude::*;
///
/// # fn main() -> PyResult<()> {
/// Python::with_gil(|py| -> PyResult<()> {
///     let list = py.eval("iter([1, 2, 3, 4])", None, None)?;
///     let numbers: PyResult<Vec<usize>> = list
///         .iter()?
///         .map(|i| i.and_then(PyAny::extract::<usize>))
///         .collect();
///     let sum: usize = numbers?.iter().sum();
///     assert_eq!(sum, 10);
///     Ok(())
/// })
/// # }
/// ```
#[repr(transparent)]
pub struct PyIterator(PyAny);
pyobject_native_type_named!(PyIterator);
pyobject_native_type_extract!(PyIterator);

impl PyIterator {
    /// Constructs a `PyIterator` from a Python iterable object.
    ///
    /// Equivalent to Python's built-in `iter` function.
    pub fn from_object(obj: &PyAny) -> PyResult<&PyIterator> {
        Self::from_object2(Py2::borrowed_from_gil_ref(&obj)).map(Py2::into_gil_ref)
    }

    pub(crate) fn from_object2<'py>(obj: &Py2<'py, PyAny>) -> PyResult<Py2<'py, PyIterator>> {
        unsafe {
            ffi::PyObject_GetIter(obj.as_ptr())
                .assume_owned_or_err(obj.py())
                .downcast_into_unchecked()
        }
    }
}

impl<'p> Iterator for &'p PyIterator {
    type Item = PyResult<&'p PyAny>;

    /// Retrieves the next item from an iterator.
    ///
    /// Returns `None` when the iterator is exhausted.
    /// If an exception occurs, returns `Some(Err(..))`.
    /// Further `next()` calls after an exception occurs are likely
    /// to repeatedly result in the same exception.
    fn next(&mut self) -> Option<Self::Item> {
        let py = self.0.py();

        match unsafe { py.from_owned_ptr_or_opt(ffi::PyIter_Next(self.0.as_ptr())) } {
            Some(obj) => Some(Ok(obj)),
            None => PyErr::take(py).map(Err),
        }
    }

    #[cfg(not(Py_LIMITED_API))]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = unsafe { ffi::PyObject_LengthHint(self.0.as_ptr(), 0) };
        (hint.max(0) as usize, None)
    }
}

impl PyTypeCheck for PyIterator {
    const NAME: &'static str = "Iterator";

    fn type_check(object: &PyAny) -> bool {
        unsafe { ffi::PyIter_Check(object.as_ptr()) != 0 }
    }
}

#[allow(deprecated)]
impl<'v> crate::PyTryFrom<'v> for PyIterator {
    fn try_from<V: Into<&'v PyAny>>(value: V) -> Result<&'v PyIterator, PyDowncastError<'v>> {
        let value = value.into();
        unsafe {
            if ffi::PyIter_Check(value.as_ptr()) != 0 {
                Ok(value.downcast_unchecked())
            } else {
                Err(PyDowncastError::new(value, "Iterator"))
            }
        }
    }

    fn try_from_exact<V: Into<&'v PyAny>>(value: V) -> Result<&'v PyIterator, PyDowncastError<'v>> {
        value.into().downcast()
    }

    #[inline]
    unsafe fn try_from_unchecked<V: Into<&'v PyAny>>(value: V) -> &'v PyIterator {
        let ptr = value.into() as *const _ as *const PyIterator;
        &*ptr
    }
}

#[cfg(test)]
mod tests {
    use super::PyIterator;
    use crate::exceptions::PyTypeError;
    use crate::gil::GILPool;
    use crate::types::{PyDict, PyList};
    use crate::{Py, PyAny, Python, ToPyObject};

    #[test]
    fn vec_iter() {
        Python::with_gil(|py| {
            let obj = vec![10, 20].to_object(py);
            let inst = obj.as_ref(py);
            let mut it = inst.iter().unwrap();
            assert_eq!(
                10_i32,
                it.next().unwrap().unwrap().extract::<'_, i32>().unwrap()
            );
            assert_eq!(
                20_i32,
                it.next().unwrap().unwrap().extract::<'_, i32>().unwrap()
            );
            assert!(it.next().is_none());
        });
    }

    #[test]
    fn iter_refcnt() {
        let (obj, count) = Python::with_gil(|py| {
            let obj = vec![10, 20].to_object(py);
            let count = obj.get_refcnt(py);
            (obj, count)
        });

        Python::with_gil(|py| {
            let inst = obj.as_ref(py);
            let mut it = inst.iter().unwrap();

            assert_eq!(
                10_i32,
                it.next().unwrap().unwrap().extract::<'_, i32>().unwrap()
            );
        });

        Python::with_gil(|py| {
            assert_eq!(count, obj.get_refcnt(py));
        });
    }

    #[test]
    fn iter_item_refcnt() {
        Python::with_gil(|py| {
            let count;
            let obj = py.eval("object()", None, None).unwrap();
            let list = {
                let _pool = unsafe { GILPool::new() };
                let list = PyList::empty(py);
                list.append(10).unwrap();
                list.append(obj).unwrap();
                count = obj.get_refcnt();
                list.to_object(py)
            };

            {
                let _pool = unsafe { GILPool::new() };
                let inst = list.as_ref(py);
                let mut it = inst.iter().unwrap();

                assert_eq!(
                    10_i32,
                    it.next().unwrap().unwrap().extract::<'_, i32>().unwrap()
                );
                assert!(it.next().unwrap().unwrap().is(obj));
                assert!(it.next().is_none());
            }
            assert_eq!(count, obj.get_refcnt());
        });
    }

    #[test]
    fn fibonacci_generator() {
        let fibonacci_generator = r#"
def fibonacci(target):
    a = 1
    b = 1
    for _ in range(target):
        yield a
        a, b = b, a + b
"#;

        Python::with_gil(|py| {
            let context = PyDict::new(py);
            py.run(fibonacci_generator, None, Some(context)).unwrap();

            let generator = py.eval("fibonacci(5)", None, Some(context)).unwrap();
            for (actual, expected) in generator.iter().unwrap().zip(&[1, 1, 2, 3, 5]) {
                let actual = actual.unwrap().extract::<usize>().unwrap();
                assert_eq!(actual, *expected)
            }
        });
    }

    #[test]
    fn int_not_iterable() {
        Python::with_gil(|py| {
            let x = 5.to_object(py);
            let err = PyIterator::from_object(x.as_ref(py)).unwrap_err();

            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]

    fn iterator_try_from() {
        Python::with_gil(|py| {
            let obj: Py<PyAny> = vec![10, 20].to_object(py).as_ref(py).iter().unwrap().into();
            let iter: &PyIterator = obj.downcast(py).unwrap();
            assert!(obj.is(iter));
        });
    }

    #[test]
    #[cfg(feature = "macros")]
    fn python_class_not_iterator() {
        use crate::PyErr;

        #[crate::pyclass(crate = "crate")]
        struct Downcaster {
            failed: Option<PyErr>,
        }

        #[crate::pymethods(crate = "crate")]
        impl Downcaster {
            fn downcast_iterator(&mut self, obj: &PyAny) {
                self.failed = Some(obj.downcast::<PyIterator>().unwrap_err().into());
            }
        }

        // Regression test for 2913
        Python::with_gil(|py| {
            let downcaster = Py::new(py, Downcaster { failed: None }).unwrap();
            crate::py_run!(
                py,
                downcaster,
                r#"
                    from collections.abc import Sequence

                    class MySequence(Sequence):
                        def __init__(self):
                            self._data = [1, 2, 3]

                        def __getitem__(self, index):
                            return self._data[index]

                        def __len__(self):
                            return len(self._data)

                    downcaster.downcast_iterator(MySequence())
                "#
            );

            assert_eq!(
                downcaster.borrow_mut(py).failed.take().unwrap().to_string(),
                "TypeError: 'MySequence' object cannot be converted to 'Iterator'"
            );
        });
    }

    #[test]
    #[cfg(feature = "macros")]
    fn python_class_iterator() {
        #[crate::pyfunction(crate = "crate")]
        fn assert_iterator(obj: &PyAny) {
            assert!(obj.downcast::<PyIterator>().is_ok())
        }

        // Regression test for 2913
        Python::with_gil(|py| {
            let assert_iterator = crate::wrap_pyfunction!(assert_iterator, py).unwrap();
            crate::py_run!(
                py,
                assert_iterator,
                r#"
                    class MyIter:
                        def __next__(self):
                            raise StopIteration

                    assert_iterator(MyIter())
                "#
            );
        });
    }

    #[test]
    #[cfg(not(Py_LIMITED_API))]
    fn length_hint_becomes_size_hint_lower_bound() {
        Python::with_gil(|py| {
            let list = py.eval("[1, 2, 3]", None, None).unwrap();
            let iter = list.iter().unwrap();
            let hint = iter.size_hint();
            assert_eq!(hint, (3, None));
        });
    }
}
