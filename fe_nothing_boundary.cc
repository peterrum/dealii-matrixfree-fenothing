#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_rhs.h>

#include <fstream>

using namespace dealii;

using VectorType = LinearAlgebra::distributed::Vector<double>;

// x > x_cutoff is the position where FE_Nothings start
// e.g. set this to 0.5 to create an internal boundary,
// 1.0 creates the reference case where the Neumann condition is applied on
// actual boundary faces
const double x_cutoff = 0.5;

namespace
{
  template <int dim, typename Number, typename VectorizedArrayType>
  void
  determine_internal_boundary_faces(
    std::vector<std::bitset<VectorizedArrayType::size()>> &     selectedFaces,
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &data,
    const std::function<
      bool(const typename dealii::DoFHandler<dim>::cell_iterator &,
           unsigned int)> &predicate)
  {
    selectedFaces.clear();
    selectedFaces.resize(data.n_inner_face_batches());

    for (unsigned int face_batch_index = 0;
         face_batch_index < data.n_inner_face_batches();
         ++face_batch_index)
      {
        const auto face_info = data.get_face_info(face_batch_index);
        for (unsigned int v = 0;
             v < data.n_active_entries_per_face_batch(face_batch_index);
             ++v)
          {
            AssertThrow(face_info.cells_interior[v] !=
                          dealii::numbers::invalid_unsigned_int,
                        ExcInternalError());
            AssertThrow(face_info.cells_exterior[v] !=
                          dealii::numbers::invalid_unsigned_int,
                        ExcInternalError());

            const auto [cell_iterator_i, face_i] =
              data.get_face_iterator(face_batch_index, v, true);
            const auto [cell_iterator_e, face_e] =
              data.get_face_iterator(face_batch_index, v, false);

            const auto hasDofs = [](auto &cell) {
              return cell->get_fe().n_dofs_per_cell() > 0;
            };

            // both cells have dofs -> this is a truly interior face and thus
            // skipped
            if (hasDofs(cell_iterator_i) and hasDofs(cell_iterator_e))
              continue;

            selectedFaces[face_batch_index][v] =
              (hasDofs(cell_iterator_i) and
               predicate(cell_iterator_i, face_i)) or
              (hasDofs(cell_iterator_e) and predicate(cell_iterator_e, face_e));

            if (selectedFaces[face_batch_index][v])
              std::cout << "BC face set" << std::endl;
          }
      }
  }
} // namespace


template <int dim>
class LaplaceOperator
{
public:
  using Number              = typename VectorType::value_type;
  using VectorizedArrayType = VectorizedArray<Number, 1>;

  /**
   * Encapsulates Laplace with Neumann boundary conditions applied to one side.
   */
  LaplaceOperator(const DoFHandler<dim> &          dof_handler,
                  const AffineConstraints<Number> &constraints,
                  const Quadrature<dim> &          quadrature)
    : constraints(constraints)
  {
    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData data;
    data.mapping_update_flags                = update_values | update_gradients;
    data.mapping_update_flags_boundary_faces = update_JxW_values;
    data.mapping_update_flags_inner_faces    = update_JxW_values;
    matrix_free.reinit(MappingQ1<dim>(),
                       dof_handler,
                       constraints,
                       hp::QCollection<dim>{quadrature, quadrature},
                       data);

    determine_internal_boundary_faces(
      internal_boundary_face_mask, matrix_free, [](auto &cell, unsigned face) {
        return std::fabs(cell->face(face)->center()[0] - x_cutoff) < 1e-8;
      });
  }


  void
  initialize_dof_vector(VectorType &vector) const
  {
    matrix_free.template initialize_dof_vector(vector);
  }

  /// linear operator
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.cell_loop(
      &LaplaceOperator::do_cell_integral_range, this, dst, src, true);
  }

  /// inhomogenous residual
  void
  residual(VectorType &dst, const VectorType &src) const
  {
    matrix_free.template loop(&LaplaceOperator::do_cell_integral_range,
                              &LaplaceOperator::do_face_integral_range,
                              &LaplaceOperator::do_boundary_integral_range,
                              this,
                              dst,
                              src,
                              true);
  }

private:
  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;
  using FEFaceIntegrator =
    FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  void
  do_cell_integral_range(
    const MatrixFree<dim, Number, VectorizedArrayType> &data,
    VectorType &                                        dst,
    const VectorType &                                  src,
    const std::pair<unsigned int, unsigned int> &       range) const
  {
    if (data.get_cell_range_category(range) == fe_nothing_index)
      return;

    FECellIntegrator integrator(data, range);
    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        integrator.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            integrator.submit_gradient(scaling * integrator.get_gradient(q), q);
          }

        integrator.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  void
  do_face_integral_range(
    const MatrixFree<dim, Number, VectorizedArrayType> &data,
    VectorType &                                        dst,
    const VectorType &                                  src,
    const std::pair<unsigned int, unsigned int> &       range) const
  {
    FEFaceIntegrator integrator_i(data, true);
    FEFaceIntegrator integrator_e(data, false);
    for (unsigned face = range.first; face < range.second; ++face)
      {
        const auto &mask = internal_boundary_face_mask[face];
        // skip face batches where none of the faces has a condition
        if (mask.none())
          continue;

        std::cout << "internal boundary encountered" << std::endl;

        integrator_i.reinit(face);
        integrator_e.reinit(face);
        // pick the integrator which has the FE with dofs
        auto &integrator =
          integrator_i.get_active_fe_index() == fe_something_index ?
            integrator_i :
            integrator_e;

        Assert(integrator.get_active_fe_index() == fe_something_index,
               ExcInternalError());

        integrator.reinit(face);

        integrator.read_dof_values(src);
        integrator.evaluate(EvaluationFlags::values);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            integrator.submit_value(
              -make_vectorized_array<VectorizedArrayType>(neumann_constant), q);
          }
        integrator.integrate(EvaluationFlags::values);
        integrator.distribute_local_to_global(dst);
      }
  }

  void
  do_boundary_integral_range(
    const MatrixFree<dim, Number, VectorizedArrayType> &data,
    VectorType &                                        dst,
    const VectorType &                                  src,
    const std::pair<unsigned int, unsigned int> &       range) const
  {
    FEFaceIntegrator integrator(data, true);

    for (unsigned face = range.first; face < range.second; ++face)
      {
        if (data.get_boundary_id(face) == 0) // should be left face?
          {
            std::cout << data.get_face_category(face).first << ","
                      << data.get_face_category(face).second << std::endl;

            std::cout << "boundary encountered" << std::endl;
            integrator.reinit(face);

            integrator.gather_evaluate(src, EvaluationFlags::values);
            for (unsigned int q = 0; q < integrator.n_q_points; ++q)
              {
                integrator.submit_value(
                  -make_vectorized_array<VectorizedArrayType>(neumann_constant),
                  q);
              }
            integrator.integrate_scatter(EvaluationFlags::values, dst);
          }
      }
  }

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  const AffineConstraints<Number> &            constraints;

  std::vector<std::bitset<VectorizedArrayType::size()>>
    internal_boundary_face_mask;

  const double scaling          = 1.0;
  const double neumann_constant = 1.0;

  const unsigned fe_something_index = 0u;
  const unsigned fe_nothing_index   = 1u;
};

template <int dim>
void
run()
{
  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0, /*colorize=*/true);
  triangulation.refine_global(2);

  DoFHandler<dim> dof_handler(triangulation);



  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        cell->set_active_fe_index(cell->center()[0] < x_cutoff ? 0 : 1);
    }

  hp::FECollection<dim> fe(FE_Q<dim>(1), FE_Nothing<dim>(1));

  dof_handler.distribute_dofs(fe);


  AffineConstraints<double> constraints;

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  constraints.reinit(locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();

  QGauss<dim> quadrature(2);

  LaplaceOperator<dim> op(dof_handler, constraints, quadrature);

  SolverControl        control;
  SolverCG<VectorType> solver_cg(control);

  VectorType b;
  VectorType x;
  op.initialize_dof_vector(b);
  op.initialize_dof_vector(x);

  op.residual(b, x);

  // solve a nonlinear problem: J*delta x = -residual
  // one solve step is enough since the problem is in fact linear
  // the final solution is x = 0 + delta x = delta x
  b *= -1.0;
  solver_cg.template solve(op, x, b, PreconditionIdentity());

  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(x, "x");
    data_out.build_patches();
    std::ofstream stream("result.vtu");
    data_out.write_vtu(stream);
  }
}


int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      run<3>();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}