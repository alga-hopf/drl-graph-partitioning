#include <iostream>
#include <vector>
#include <cassert>

#include <scotch.h>

#ifdef __cplusplus
extern "C" {
#endif

  void WRAPPER_SCOTCH_graphPart(int n, int* ptr, int* ind, int* part) {
    SCOTCH_Graph g;
    SCOTCH_graphInit(&g);
    std::vector<SCOTCH_Num> ptr_nodiag(n+1), ind_nodiag(ptr[n]-ptr[0]);
    int nnz_nodiag = 0;
    ptr_nodiag[0] = 0;
    for (int i=0; i<n; i++) { // remove diagonal elements
      for (int j=ptr[i]; j<ptr[i+1]; j++)
        if (ind[j] != i)
          ind_nodiag[nnz_nodiag++] = ind[j];
      ptr_nodiag[i+1] = nnz_nodiag;
    }
    int ierr = SCOTCH_graphBuild
      (&g, 0, n, ptr_nodiag.data(), nullptr, nullptr, nullptr,
       ptr_nodiag[n], ind_nodiag.data(), nullptr);
    if (ierr)
      std::cerr << "# ERROR: Scotch failed to build graph." << std::endl;
    assert(SCOTCH_graphCheck(&g) == 0);
    SCOTCH_Strat strategy;
    ierr = SCOTCH_stratInit(&strategy);
    if (ierr)
      std::cerr << "# ERROR: Scotch failed to build graph." << std::endl;
    std::vector<SCOTCH_Num> p(n);
    SCOTCH_graphPart(&g, 2, &strategy, p.data());
    std::copy(p.begin(), p.end(), part);
    SCOTCH_graphExit(&g);
    SCOTCH_stratExit(&strategy);
  }

  void WRAPPER_SCOTCH_graphOrder(int n, int* ptr, int* ind, int* perm) {
    SCOTCH_Graph g;
    SCOTCH_graphInit(&g);
    std::vector<SCOTCH_Num> ptr_nodiag(n+1), ind_nodiag(ptr[n]-ptr[0]);
    int nnz_nodiag = 0;
    ptr_nodiag[0] = 0;
    for (int i=0; i<n; i++) { // remove diagonal elements
      for (int j=ptr[i]; j<ptr[i+1]; j++)
        if (ind[j] != i)
          ind_nodiag[nnz_nodiag++] = ind[j];
      ptr_nodiag[i+1] = nnz_nodiag;
    }
    int ierr = SCOTCH_graphBuild
      (&g, 0, n, ptr_nodiag.data(), nullptr, nullptr, nullptr,
       ptr_nodiag[n], ind_nodiag.data(), nullptr);
    if (ierr)
      std::cerr << "# ERROR: Scotch failed to build graph." << std::endl;
    assert(SCOTCH_graphCheck(&g) == 0);
    SCOTCH_Strat strategy;
    ierr = SCOTCH_stratInit(&strategy);
    if (ierr)
      std::cerr << "# ERROR: Scotch failed to build graph." << std::endl;
    SCOTCH_Num nbsep;
    std::vector<SCOTCH_Num> p(n), pi(n), sizes(n+1), tree(n);
    ierr = SCOTCH_graphOrder
      (&g, &strategy, p.data(), pi.data(),
       &nbsep, sizes.data(), tree.data());
    if (ierr)
      std::cerr << "# ERROR: SCOTCH_graphOrder faile with ierr="
                << ierr << std::endl;
    std::copy(pi.begin(), pi.end(), perm);
    SCOTCH_graphExit(&g);
    SCOTCH_stratExit(&strategy);
  }

#ifdef __cplusplus
}
#endif
