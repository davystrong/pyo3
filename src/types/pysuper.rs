use crate::instance::Py2;
use crate::types::any::PyAnyMethods;
use crate::types::PyType;
use crate::{ffi, PyTypeInfo};
use crate::{PyAny, PyResult};

/// Represents a Python `super` object.
///
/// This type is immutable.
#[repr(transparent)]
pub struct PySuper(PyAny);

pyobject_native_type_core!(
    PySuper,
    pyobject_native_static_type_object!(ffi::PySuper_Type)
);

impl PySuper {
    /// Constructs a new super object. More read about super object: [docs](https://docs.python.org/3/library/functions.html#super)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pyo3::prelude::*;
    ///
    /// #[pyclass(subclass)]
    /// struct BaseClass {
    ///     val1: usize,
    /// }
    ///
    /// #[pymethods]
    /// impl BaseClass {
    ///     #[new]
    ///     fn new() -> Self {
    ///         BaseClass { val1: 10 }
    ///     }
    ///
    ///     pub fn method(&self) -> usize {
    ///         self.val1
    ///     }
    /// }
    ///
    /// #[pyclass(extends=BaseClass)]
    /// struct SubClass {}
    ///
    /// #[pymethods]
    /// impl SubClass {
    ///     #[new]
    ///     fn new() -> (Self, BaseClass) {
    ///         (SubClass {}, BaseClass::new())
    ///     }
    ///
    ///     fn method(self_: &PyCell<Self>) -> PyResult<&PyAny> {
    ///         let super_ = self_.py_super()?;
    ///         super_.call_method("method", (), None)
    ///     }
    /// }
    /// ```
    pub fn new<'py>(ty: &'py PyType, obj: &'py PyAny) -> PyResult<&'py PySuper> {
        Self::new2(
            Py2::borrowed_from_gil_ref(&ty),
            Py2::borrowed_from_gil_ref(&obj),
        )
        .map(Py2::into_gil_ref)
    }

    pub(crate) fn new2<'py>(
        ty: &Py2<'py, PyType>,
        obj: &Py2<'py, PyAny>,
    ) -> PyResult<Py2<'py, PySuper>> {
        Py2::borrowed_from_gil_ref(&PySuper::type_object(ty.py()))
            .call1((ty, obj))
            .map(|any| {
                // Safety: super() always returns instance of super
                unsafe { any.downcast_into_unchecked() }
            })
    }
}
