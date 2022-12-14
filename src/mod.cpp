#include <pybind11/pybind11.h> // Must be the first include.
#include <pybind11/attr.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <atomic>
#include <iostream>
#include "operation.h"
#include "Matrix.h"
using namespace std;

#ifdef __GNUG__
#define PYTHON_WRAPPER_VISIBILITY __attribute__((visibility("hidden")))
#else
#define PYTHON_WRAPPER_VISIBILITY
#endif

namespace python
{

    struct calltag
    {
    };

} /* end namespace python */

namespace pybind11
{
    namespace detail
    {

        template <>
        struct process_attribute<python::calltag>
            : process_attribute_default<python::calltag>
        {

            static void postcall(function_call &call, handle &)
            {
                mark(call);
            }

        private:
            static void mark(function_call const &call)
            {
                object o = module::import("blas");
                std::string name = std::string(str(call.func.scope.attr("__name__")));
                name += std::string(call.func.name) + "_post_count";
                const size_t count = getattr(o, name.c_str(), int_(0)).cast<size_t>() + 1;
                setattr(o, name.c_str(), int_(count));
            }
        };

    } /* end namespace detail */
} /* end namespace pybind11 */

namespace python
{

    /**
     * Helper template for pybind11 class wrappers.
     */
    template <
        class Wrapper, class Wrapped, class Holder = std::unique_ptr<Wrapped>, class WrappedBase = Wrapped>
    /*
     * Use CRTP to detect type error during compile time.
     */
    class
        PYTHON_WRAPPER_VISIBILITY
            WrapBase
    {

    public:
        using wrapper_type = Wrapper;
        using wrapped_type = Wrapped;
        using wrapped_base_type = WrappedBase;
        using holder_type = Holder;
        using root_base_type = WrapBase<
            wrapper_type, wrapped_type, holder_type, wrapped_base_type>;
        using class_ = typename std::conditional_t<
            std::is_same<Wrapped, WrappedBase>::value, pybind11::class_<wrapped_type, holder_type>, pybind11::class_<wrapped_type, wrapped_base_type, holder_type>>;

        static wrapper_type &commit(pybind11::module &mod)
        {
            static wrapper_type derived(mod);
            return derived;
        }

        static wrapper_type &commit(pybind11::module &mod, char const *pyname, char const *pydoc)
        {
            static wrapper_type derived(mod, pyname, pydoc);
            return derived;
        }

        WrapBase() = delete;
        WrapBase(WrapBase const &) = default;
        WrapBase(WrapBase &&) = delete;
        WrapBase &operator=(WrapBase const &) = default;
        WrapBase &operator=(WrapBase &&) = delete;
        ~WrapBase() = default;

#define DECL_PYBIND_CLASS_METHOD_UNTAGGED(METHOD)                             \
    template <class... Args> /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    wrapper_type &METHOD(Args &&...args)                                      \
    {                                                                         \
        m_cls.METHOD(std::forward<Args>(args)...);                            \
        return *static_cast<wrapper_type *>(this);                            \
    }

#define DECL_PYBIND_CLASS_METHOD_TAGGED(METHOD)                               \
    template <class... Args> /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    wrapper_type &METHOD##_tagged(Args &&...args)                             \
    {                                                                         \
        m_cls.METHOD(std::forward<Args>(args)..., calltag());                 \
        return *static_cast<wrapper_type *>(this);                            \
    }

#define DECL_PYBIND_CLASS_METHOD(METHOD)      \
    DECL_PYBIND_CLASS_METHOD_UNTAGGED(METHOD) \
    DECL_PYBIND_CLASS_METHOD_TAGGED(METHOD)

        DECL_PYBIND_CLASS_METHOD(def)
        DECL_PYBIND_CLASS_METHOD(def_static)

        DECL_PYBIND_CLASS_METHOD(def_readwrite)
        DECL_PYBIND_CLASS_METHOD(def_readonly)
        DECL_PYBIND_CLASS_METHOD(def_readwrite_static)
        DECL_PYBIND_CLASS_METHOD(def_readonly_static)

        DECL_PYBIND_CLASS_METHOD(def_property)
        DECL_PYBIND_CLASS_METHOD(def_property_static)
        DECL_PYBIND_CLASS_METHOD(def_property_readonly)
        DECL_PYBIND_CLASS_METHOD(def_property_readonly_static)

#undef DECL_PYBIND_CLASS_METHOD
#undef DECL_PYBIND_CLASS_METHOD_TAGGED
#undef DECL_PYBIND_CLASS_METHOD_UNTAGGED

        class_ &cls()
        {
            return m_cls;
        }

    protected:
        template <typename... Extra>
        WrapBase(pybind11::module &mod, char const *pyname, char const *pydoc, const Extra &...extra)
            : m_cls(mod, pyname, pydoc, extra...)
        {
        }

    private:
        class_ m_cls;

    }; /* end class WrapBase */

    /**
     * Wrapper for the Matrix class.
     */
    class
        PYTHON_WRAPPER_VISIBILITY
            WrapMatrix
        : public WrapBase<WrapMatrix, Matrix>
    {

        friend root_base_type;

        WrapMatrix(pybind11::module &mod, char const *pyname, char const *pydoc)
            : root_base_type(mod, pyname, pydoc, pybind11::buffer_protocol())
        {

            namespace py = pybind11;

            (*this)
                // The Python constructor will be counted!
                .def_tagged(py::init<size_t, size_t>())
                .def_property
                (
                    "array"
                    ,[](Matrix &m)
                    { return py::array_t<double>({m.rows(), m.cols()},
                     {sizeof(double) * m.cols(), sizeof(double)}, m.data(), py::cast(m)); },
                    nullptr
                )
                .def_property_readonly("nrow", &Matrix::rows)
                .def_property_readonly("ncol", &Matrix::cols)
                .def("__eq__", [](Matrix const &self, Matrix const &other)
                     { return self == other; })
                .def(
                    "__getitem__", [](Matrix const &self, std::tuple<size_t, size_t> idx)
                    { return self(std::get<0>(idx), std::get<1>(idx)); })
                .def(
                    "__setitem__", [](Matrix &self, std::tuple<size_t, size_t> idx, double value)
                    { return self(std::get<0>(idx), std::get<1>(idx)) = value; });

            mod.def("multiply_naive", &multiply_naive, "Matrix multiply with naive method.");
            mod.def("multiply_tile", &multiply_tile, "Matrix multiply with tile method.");
            mod.def("multiply_mkl", &multiply_mkl, "Matrix multiply with mkl method.");
            mod.def("lu_Decomposition_naive", &lu_Decomposition_naive, "naive LU decomposition");
            mod.def("lu_Decomposition_mkl", &lu_Decomposition_mkl, "LAPACK LU decomposition");
            mod.def("qr_Decomposition_naive", &qr_Decomposition_naive, "navie QR decomposition");
            mod.def("qr_Decomposition_mkl", &qr_Decomposition_mkl, "LAPACK QR decomposition");
            mod.def("naive_lu_solver", &naive_lu_solver, "LU solve linear equation");
            mod.def("mkl_lu_solver", &mkl_lu_solver, "LU solve linear equation");
            
        }

    }; /* end class WrapMatrix */

} /* end namespace python */

PYBIND11_MODULE(blas, mod)
{
    python::WrapMatrix::commit(mod, "Matrix", "Matrix");
}