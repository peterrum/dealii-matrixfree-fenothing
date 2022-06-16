#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

constexpr unsigned int fe_index_valid   = 0;
constexpr unsigned int fe_index_nothing = 1;

using namespace dealii;

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
      cell->set_active_fe_index(fe_index_valid);
    else
      cell->set_active_fe_index(fe_index_nothing);

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
    [&](const auto &matrix_free, auto &, auto &, const auto range) {
      const auto category = matrix_free.get_cell_range_category(range);

      if (category != fe_index_valid)
        return;

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
    [&](const auto &matrix_free, auto &, auto &, const auto range) {
      const auto category = matrix_free.get_face_range_category(range);

      if (static_cast<unsigned int>(category.first == fe_index_valid) +
            static_cast<unsigned int>(category.second == fe_index_valid) !=
          1)
        return;

      std::cout << "face:" << std::endl;

      FEFaceIntegrator phi(matrix_free, category.first == fe_index_valid);

      for (unsigned int face = range.first; face < range.second; ++face)
        {
          phi.reinit(face);

          for (const auto q : phi.quadrature_point_indices())
            print_points(phi.quadrature_point(q),
                         matrix_free.n_active_entries_per_face_batch(face));
        }
    };

  matrix_free.template cell_loop<VectorType, VectorType>(cell_operation,
                                                         dst,
                                                         src);
  std::cout << std::endl;
  matrix_free.template loop<VectorType, VectorType>(
    cell_operation, face_operation, face_operation, dst, src);
}

int
main()
{
  test<2>(1);
}