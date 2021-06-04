#include <memory>
#include <limits>
#include <iostream>
#include <vector>

#if AMDIDXSIZE==64
  typedef int64_t AMDInt;
#else
  typedef int32_t AMDInt;
#endif

#define FC_GLOBAL(name,NAME) name##_
#define AMDBAR_FC FC_GLOBAL(amdbar,AMDBAR)

#ifdef __cplusplus
extern "C" {
#endif

  void AMDBAR_FC(AMDInt* N, AMDInt* PE, AMDInt* IW, AMDInt* LEN,
                 AMDInt* IWLEN, AMDInt* PFREE, AMDInt* NV,
                 AMDInt* NEXT, AMDInt* LAST, AMDInt* HEAD,
                 AMDInt* ELEN, AMDInt* DEGREE, AMDInt* NCMPA,
                 AMDInt* W, AMDInt* IOVFLO);

  /*
   * Input to this routine should be 0-based
   */
  void WRAPPER_amd(AMDInt n, AMDInt* xadj, AMDInt* adjncy,
                   AMDInt* perm, AMDInt* iperm) {
    AMDInt iovflo = std::numeric_limits<AMDInt>::max();
    AMDInt ncmpa = 0;
    AMDInt iwsize = 4*n;
    AMDInt nnz = xadj[n];

    std::vector<AMDInt> ptr(n+1);
    std::vector<AMDInt> ind(nnz);
    for (AMDInt i=0; i<=n; i++) ptr[i] = xadj[i] + 1;
    for (AMDInt i=0; i<nnz; i++) ind[i] = adjncy[i] + 1;

    std::unique_ptr<AMDInt[]> iwork    // iwsize
      (new AMDInt[iwsize + 4*n + n+1 + nnz + n + 1]);
    auto vtxdeg = iwork.get() + iwsize; // n
    auto qsize  = vtxdeg + n;     // n
    auto ecforw = qsize + n;      // n
    auto marker = ecforw + n;     // n
    auto nvtxs  = marker + n;     // n+1
    auto rowind = nvtxs + n+1;    // nnz + n + 1
    for (AMDInt i=0; i<n; i++)
      nvtxs[i] = ptr[i+1] - ptr[i];
    for (AMDInt i=0; i<nnz; i++)
      rowind[i] = ind[i];
    AMDInt pfree = ptr[n-1] + nvtxs[n-1];
    AMDInt iwlen = pfree + n;
    AMDBAR_FC(&n, ptr.data(), rowind, nvtxs, &iwlen, &pfree, qsize, ecforw,
              perm, iwork.get(), iperm, vtxdeg, &ncmpa, marker, &iovflo);

    for (AMDInt i=0; i<n; i++) perm[i]--;
    for (AMDInt i=0; i<n; i++) iperm[i]--;
  }

#ifdef __cplusplus
}
#endif
