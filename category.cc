#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

template <int dim>
void
test(const unsigned int n_refinements)
{
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {3, 1},
                                            {0.0, 0.0},
                                            {3.0, 1.0});
  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(1));

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData data;
  data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
  data.mapping_update_flags  = update_quadrature_points;
  data.mapping_update_flags_boundary_faces = update_quadrature_points;
  data.mapping_update_flags_inner_faces    = update_quadrature_points;
  data.hold_all_faces_to_owned_cells       = true;

  std::vector<unsigned int> cell_vectorization_category(tria.n_active_cells());
  for (unsigned int i = 0; i < cell_vectorization_category.size(); ++i)
    cell_vectorization_category[i] = i;

  data.cell_vectorization_category = cell_vectorization_category;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(MappingQ1<dim>(),
                     dof_handler,
                     AffineConstraints<Number>(),
                     QGauss<dim>(2),
                     data);
}

/**
 * mpirun -np 3 ./category
 */
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  AssertDimension(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 3);

  test<2>(0);
}