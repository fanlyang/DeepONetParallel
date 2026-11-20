Build in fanyoung/dealii-deeponet:latest on a matching platform. Inside the mounted project directory:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/opt/dealii-9.7.1-codim/lib/cmake/deal.II
    cmake --build build --parallel 2
    mpiexec -n 4 ./build/torus_patch_mpi --patch-map patch_map.txt

patch_map.txt is whitespace-separated, no header, zero-based DoF and patch IDs; use the finest-level deal.II MG DoF numbering. Exactly 16384 pairs, 1024 per patch. The program checks coverage/counts but cannot infer whether a supplied file uses the same numbering; cross-check coordinates when importing an independently exported map. With no flag, a geometric 4x4 angular tiling is used as a demo; its indexing is not a model checkpoint mapping.

The program builds A on rank 0, distributes full sparse block rows, and exchanges only required remote vector entries. Rank 0 retains the assembled matrix and gathers the result solely for the serial reference check. This is a single-level matrix-vector verification executable, not CG/MG, a distributed assembly benchmark, or a solver timing benchmark. No PETSc API or Python is used. The installed deal.II library may itself have optional packages enabled.

Test with 1, 2, 4, 8 and 16 ranks as slots permit. Exit 0 / PASS means the relative L2 error against serial deal.II vmult is <= 1e-12. With Open MPI running as root in Docker, add --allow-run-as-root if needed.

No compilation or MPI convergence claims: the authoring environment lacks Docker and the deal.II development installation.
