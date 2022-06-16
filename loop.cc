#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

namespace dealii
{
  namespace MatrixFreeTools
  {
    class BirthAndDeath
    {
    public:
      static constexpr unsigned int fe_index_valid   = 0;
      static constexpr unsigned int fe_index_nothing = 1;

      template <typename VectorTypeOut,
                typename VectorTypeIn,
                int dim,
                typename Number,
                typename VectorizedArrayType>
      static void
      cell_loop(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
                const std::function<void(
                  const MatrixFree<dim, Number, VectorizedArrayType> &,
                  VectorTypeOut &,
                  const VectorTypeIn &,
                  const std::pair<unsigned int, unsigned int>)> &cell_operation,
                VectorTypeOut &                                  dst,
                const VectorTypeIn &                             src)
      {
        const auto ebd_cell_operation = [&](const auto &matrix_free,
                                            auto &      dst,
                                            const auto &src,
                                            const auto  range) {
          const auto category = matrix_free.get_cell_range_category(range);

          if (category != MatrixFreeTools::BirthAndDeath::fe_index_valid)
            return;

          cell_operation(matrix_free, dst, src, range);
        };

        matrix_free.template cell_loop<VectorTypeOut, VectorTypeIn>(
          ebd_cell_operation, dst, src);
      }

      void
      loop()
      {}
    };
  } // namespace MatrixFreeTools
} // namespace dealii

template <int dim>
void
test(const unsigned int n_refinements)
{
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = Vector<Number>;
  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;
  using FEFaceIntegrator =
    FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler(tria);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->center()[1] < 0.5)
      cell->set_active_fe_index(MatrixFreeTools::BirthAndDeath::fe_index_valid);
    else
      cell->set_active_fe_index(
        MatrixFreeTools::BirthAndDeath::fe_index_nothing);

  dof_handler.distribute_dofs(
    hp::FECollection<dim>(FE_Q<dim>(1), FE_Nothing<dim>(1)));

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData data;
  data.mapping_update_flags                = update_quadrature_points;
  data.mapping_update_flags_boundary_faces = update_quadrature_points;
  data.mapping_update_flags_inner_faces    = update_quadrature_points;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(MappingQ1<dim>(),
                     dof_handler,
                     AffineConstraints<Number>(),
                     QGauss<dim>(2),
                     data);

  VectorType dst, src;
  matrix_free.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(src);

  const auto print_points = [](const auto points, const unsigned int n_points) {
    for (unsigned int i = 0; i < n_points; ++i)
      {
        Point<dim> point;

        for (unsigned int d = 0; d < dim; ++d)
          point[d] = points[d][i];
        std::cout << std::fixed << std::setprecision(5) << point << "    ";
      }
    std::cout << std::endl;
  };

  const auto cell_operation =
    [&](const auto &matrix_free, auto &, const auto &, const auto range) {
      std::cout << "cell:" << std::endl;

      FECellIntegrator phi(matrix_free);

      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);

          for (const auto q : phi.quadrature_point_indices())
            print_points(phi.quadrature_point(q),
                         matrix_free.n_active_entries_per_cell_batch(cell));
        }
    };

  const auto face_operation =
    [&](const auto &, auto &, const auto &, const auto) {
      // nothing to do here but in the DG case
    };

  const auto boundary_operation = [&](const auto &matrix_free,
                                      auto &,
                                      const auto &,
                                      const auto range,
                                      const bool is_interior_face) {
    std::cout << "face:" << std::endl;

    FEFaceIntegrator phi(matrix_free, is_interior_face);

    for (unsigned int face = range.first; face < range.second; ++face)
      {
        phi.reinit(face);

        for (const auto q : phi.quadrature_point_indices())
          print_points(phi.quadrature_point(q),
                       matrix_free.n_active_entries_per_face_batch(face));
      }
  };

  const auto ebd_cell_operation =
    [&](const auto &matrix_free, auto &dst, const auto &src, const auto range) {
      const auto category = matrix_free.get_cell_range_category(range);

      if (category != MatrixFreeTools::BirthAndDeath::fe_index_valid)
        return;

      cell_operation(matrix_free, dst, src, range);
    };

  const auto ebd_internal_or_boundary_face_operation =
    [&](const auto &matrix_free, auto &dst, const auto &src, const auto range) {
      const auto category = matrix_free.get_face_range_category(range);

      const unsigned int type =
        static_cast<unsigned int>(
          category.first == MatrixFreeTools::BirthAndDeath::fe_index_valid) +
        static_cast<unsigned int>(
          category.second == MatrixFreeTools::BirthAndDeath::fe_index_valid);

      if (type == 1) // boundary face
        boundary_operation(matrix_free,
                           dst,
                           src,
                           range,
                           category.first ==
                             MatrixFreeTools::BirthAndDeath::fe_index_valid);
      else if (type == 2) // internal face
        face_operation(matrix_free, dst, src, range);
    };


  MatrixFreeTools::BirthAndDeath::template cell_loop<VectorType,
                                                     VectorType,
                                                     dim,
                                                     Number,
                                                     VectorizedArrayType>(
    matrix_free, cell_operation, dst, src);
  std::cout << std::endl;
  matrix_free.template loop<VectorType, VectorType>(
    ebd_cell_operation,
    ebd_internal_or_boundary_face_operation,
    ebd_internal_or_boundary_face_operation,
    dst,
    src);
}

int
main()
{
  test<2>(1);
}