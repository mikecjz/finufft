/* Tests opts.gpu_no_cropping: the type-1 variant that skips the crop of the
   deconvolved fine grid and emits the whole nf1*nf2*nf3 upsampled grid.

   Three things are checked, for dim = 1, 2, 3:

   1. cufinufft_get_out_modes reports the fine grid, not the requested modes.

   2. CENTRAL-MODE EQUIVALENCE (the real regression guard). The central
      ms*mt*mu block of the uncropped output must equal a normal cropped plan's
      output. Both runs share the same fw, the same fwkerhalf entries and the
      same index arithmetic, so agreement is expected to be exact; a small
      tolerance is allowed only for nondeterministic spread atomics.

   3. OUTER-BAND BEHAVIOUR. Against a direct DFT over the full grid, the error
      is ~tol in the central band and degrades towards O(1) at the fine-grid
      Nyquist. That degradation is inherent, not a bug: deconvolution recovers
      f_k + sum_{m!=0} [phihat(xi+m)/phihat(xi)] f_{k+m*nf}, and the aliasing
      ratio grows from ~tol at |k|=ms/2 to ~1 at |k|=nf/2 (where a mode's alias
      is itself). So the test asserts tol ONLY on the central band, and prints
      the error-vs-frequency curve. Asserting tol on the outer band would fail
      by design.

   Also checks that modeord=1 gives the same grid as modeord=0 up to an
   fftshift, which for the uncropped output is the raw deconvolved fw.
*/

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cufinufft.h>
#include <finufft_common/common.h>

#include "../utils/dirft1d.hpp"
#include "../utils/dirft2d.hpp"
#include "../utils/dirft3d.hpp"
#include "../utils/norms.hpp"
#include <cufinufft/contrib/helper_cuda.h>

using ::finufft::common::PI;
using Cpx = thrust::complex<double>;

// Run a type-1 double-precision transform; returns the output and, via
// out_modes, the plan's effective output extent.
static int run_type1(int dim, const int64_t n_modes[3], int M,
                     const thrust::device_vector<double> &d_x,
                     const thrust::device_vector<double> &d_y,
                     const thrust::device_vector<double> &d_z,
                     const thrust::device_vector<Cpx> &d_c, double tol, int iflag,
                     int no_cropping, int modeord, thrust::host_vector<Cpx> &out,
                     int64_t out_modes[3]) {
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_no_cropping = no_cropping;
  opts.modeord         = modeord;

  cufinufft_plan plan;
  int ier = cufinufft_makeplan(1, dim, n_modes, iflag, 1, tol, &plan, &opts);
  if (ier != 0) {
    std::cout << "  makeplan failed, ier=" << ier << "\n";
    return ier;
  }

  ier = cufinufft_get_out_modes(plan, out_modes);
  if (ier != 0) {
    std::cout << "  get_out_modes failed, ier=" << ier << "\n";
    cufinufft_destroy(plan);
    return ier;
  }

  int64_t ntot = 1;
  for (int d = 0; d < dim; ++d) ntot *= out_modes[d];

  ier = cufinufft_setpts(plan, M, (double *)d_x.data().get(),
                         dim > 1 ? (double *)d_y.data().get() : nullptr,
                         dim > 2 ? (double *)d_z.data().get() : nullptr, 0, nullptr,
                         nullptr, nullptr);
  if (ier != 0) {
    std::cout << "  setpts failed, ier=" << ier << "\n";
    cufinufft_destroy(plan);
    return ier;
  }

  thrust::device_vector<Cpx> d_fk(ntot);
  ier = cufinufft_execute(plan, (cuDoubleComplex *)d_c.data().get(),
                          (cuDoubleComplex *)d_fk.data().get());
  cufinufft_destroy(plan);
  if (ier != 0) {
    std::cout << "  execute failed, ier=" << ier << "\n";
    return ier;
  }
  out = d_fk;
  return 0;
}

static int run_dim(int dim, int N, int M, double tol) {
  const int iflag = 1;
  std::cout << "\n=== dim=" << dim << " N=" << N << " M=" << M << " tol=" << tol
            << " ===\n";

  const int64_t n_modes[3] = {N, dim > 1 ? N : 1, dim > 2 ? N : 1};

  std::default_random_engine eng(7);
  std::uniform_real_distribution<double> dist11(-1, 1);
  auto randm11 = [&]() {
    return dist11(eng);
  };

  thrust::host_vector<double> x(M), y(M), z(M);
  thrust::host_vector<Cpx> c(M);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * randm11();
    y[j] = PI * randm11();
    z[j] = PI * randm11();
    c[j] = Cpx(randm11(), randm11());
  }
  thrust::device_vector<double> d_x = x, d_y = y, d_z = z;
  thrust::device_vector<Cpx> d_c    = c;

  // --- cropped reference and uncropped run, both modeord=0 ------------------
  thrust::host_vector<Cpx> fk_crop, fk_full;
  int64_t m_crop[3], m_full[3];
  if (run_type1(dim, n_modes, M, d_x, d_y, d_z, d_c, tol, iflag, 0, 0, fk_crop, m_crop))
    return 1;
  if (run_type1(dim, n_modes, M, d_x, d_y, d_z, d_c, tol, iflag, 1, 0, fk_full, m_full))
    return 1;

  // 1. extents
  for (int d = 0; d < dim; ++d) {
    if (m_crop[d] != n_modes[d]) {
      std::cout << "  FAIL: cropped extent[" << d << "]=" << m_crop[d] << " != " << N
                << "\n";
      return 1;
    }
    if (m_full[d] <= m_crop[d] || (m_full[d] & 1)) {
      std::cout << "  FAIL: uncropped extent[" << d << "]=" << m_full[d]
                << " is not an even fine-grid size > " << m_crop[d] << "\n";
      return 1;
    }
  }
  std::cout << "  modes (" << m_crop[0] << "," << m_crop[1] << "," << m_crop[2]
            << ") -> fine grid (" << m_full[0] << "," << m_full[1] << "," << m_full[2]
            << ")\n";

  // 2. central-block equivalence. modeord=0 puts frequency f at index
  // f + extent/2, so the cropped block sits at offset (nf-ms)/2 per dim.
  const int64_t off[3] = {(m_full[0] - m_crop[0]) / 2, (m_full[1] - m_crop[1]) / 2,
                          (m_full[2] - m_crop[2]) / 2};
  double maxdiff = 0, maxabs = 0;
  for (int64_t k3 = 0; k3 < m_crop[2]; ++k3)
    for (int64_t k2 = 0; k2 < m_crop[1]; ++k2)
      for (int64_t k1 = 0; k1 < m_crop[0]; ++k1) {
        const int64_t ic = k1 + m_crop[0] * (k2 + m_crop[1] * k3);
        const int64_t if_ = (k1 + off[0]) + m_full[0] * ((k2 + off[1]) +
                                                         m_full[1] * (k3 + off[2]));
        maxdiff = std::max(maxdiff, thrust::abs(fk_crop[ic] - fk_full[if_]));
        maxabs  = std::max(maxabs, thrust::abs(fk_crop[ic]));
      }
  const double relcentral = maxdiff / maxabs;
  std::cout << "  central-block vs cropped plan: max rel diff = " << relcentral << "\n";
  if (!(relcentral < 1e-14)) {
    std::cout << "  FAIL: central modes should be identical to the cropped plan\n";
    return 1;
  }

  // 3. accuracy vs a direct DFT over the whole fine grid.
  const int64_t ntot = m_full[0] * m_full[1] * m_full[2];
  thrust::host_vector<Cpx> Ft(ntot);
  if (dim == 1)
    dirft1d1<int64_t>(int64_t(M), x, c, iflag, m_full[0], Ft);
  else if (dim == 2)
    dirft2d1<int64_t>(int64_t(M), x, y, c, iflag, m_full[0], m_full[1], Ft);
  else
    dirft3d1<int64_t>(int64_t(M), x, y, z, c, iflag, m_full[0], m_full[1], m_full[2], Ft);

  const double nrm = infnorm(ntot, Ft);

  // Bucket the error by max-norm frequency radius, normalised so that 1.0 is
  // the crop edge (|k| = ms/2) and m_full/m_crop is the fine-grid Nyquist.
  constexpr int NB = 10;
  std::vector<double> berr(NB, 0.0);
  const double rmax = double(m_full[0]) / double(m_crop[0]);
  double central_err = 0.0;
  for (int64_t i = 0; i < ntot; ++i) {
    const int64_t k1 = i % m_full[0];
    const int64_t k2 = (i / m_full[0]) % m_full[1];
    const int64_t k3 = i / (m_full[0] * m_full[1]);
    // signed frequency, relative to the crop half-width in each dim
    double r = std::abs(double(k1 - m_full[0] / 2)) / (0.5 * m_crop[0]);
    if (dim > 1)
      r = std::max(r, std::abs(double(k2 - m_full[1] / 2)) / (0.5 * m_crop[1]));
    if (dim > 2)
      r = std::max(r, std::abs(double(k3 - m_full[2] / 2)) / (0.5 * m_crop[2]));
    const double e = thrust::abs(Ft[i] - fk_full[i]) / nrm;
    int b          = int(r / rmax * NB);
    if (b >= NB) b = NB - 1;
    berr[b] = std::max(berr[b], e);
    if (r <= 1.0) central_err = std::max(central_err, e);
  }

  std::cout << "  rel err vs direct DFT, by |k|/(ms/2):\n";
  for (int b = 0; b < NB; ++b)
    std::cout << "    [" << std::fixed << std::setprecision(2) << (b * rmax / NB) << ", "
              << ((b + 1) * rmax / NB) << ")  " << std::scientific
              << std::setprecision(2) << berr[b]
              << (b * rmax / NB < 1.0 ? "" : "  (outer band: aliased by design)") << "\n";
  std::cout << std::scientific << std::setprecision(3)
            << "  central band (|k| <= ms/2) max rel err = " << central_err << "\n";

  // Only the central band carries the requested tolerance.
  const double checktol = 10 * tol;
  if (!(central_err < checktol)) {
    std::cout << "  FAIL: central band err " << central_err << " >= " << checktol << "\n";
    return 1;
  }

  // 4. modeord=1 must be the fftshift of modeord=0 (ie the raw deconvolved fw).
  thrust::host_vector<Cpx> fk_fft;
  int64_t m_fft[3];
  if (run_type1(dim, n_modes, M, d_x, d_y, d_z, d_c, tol, iflag, 1, 1, fk_fft, m_fft))
    return 1;
  // The two runs use separate plans, so their fw differ at roundoff level (the
  // spreader's atomics are not order-deterministic). In the central band the
  // deconvolution is O(1) and that stays at roundoff, so the check is tight
  // there. Further out, 1/phihat has a large dynamic range (docs/trouble.rst)
  // which amplifies that roundoff -- and compounds per dimension, since the
  // separable product gives (1/phihat)^dim at a grid corner. So the outer band
  // gets a loose bound, still tight enough to catch any real mapping error,
  // which would show up as an O(1) difference.
  double shiftdiff_c = 0, shiftdiff_o = 0;
  for (int64_t k3 = 0; k3 < m_full[2]; ++k3)
    for (int64_t k2 = 0; k2 < m_full[1]; ++k2)
      for (int64_t k1 = 0; k1 < m_full[0]; ++k1) {
        const int64_t s1 = (k1 + m_full[0] / 2) % m_full[0];
        const int64_t s2 = (k2 + m_full[1] / 2) % m_full[1];
        const int64_t s3 = (k3 + m_full[2] / 2) % m_full[2];
        const int64_t a  = k1 + m_full[0] * (k2 + m_full[1] * k3);
        const int64_t b  = s1 + m_full[0] * (s2 + m_full[1] * s3);
        const double d   = thrust::abs(fk_full[a] - fk_fft[b]) / nrm;
        double r = std::abs(double(k1 - m_full[0] / 2)) / (0.5 * m_crop[0]);
        if (dim > 1)
          r = std::max(r, std::abs(double(k2 - m_full[1] / 2)) / (0.5 * m_crop[1]));
        if (dim > 2)
          r = std::max(r, std::abs(double(k3 - m_full[2] / 2)) / (0.5 * m_crop[2]));
        if (r <= 1.0)
          shiftdiff_c = std::max(shiftdiff_c, d);
        else
          shiftdiff_o = std::max(shiftdiff_o, d);
      }
  std::cout << "  modeord=1 vs fftshift(modeord=0): central " << shiftdiff_c
            << ", outer " << shiftdiff_o << "\n";
  if (!(shiftdiff_c < 1e-13) || !(shiftdiff_o < 1e-5)) {
    std::cout << "  FAIL: modeord=1 is not the fftshift of modeord=0\n";
    return 1;
  }

  std::cout << "  dim=" << dim << " PASS\n";
  return 0;
}

// The fk batch stride also switches to the fine-grid extent, so check a
// many-transform plan: with strengths c and 2c, the second output block must be
// exactly twice the first.
static int run_batched(int dim, int N, int M, double tol) {
  const int iflag = 1, ntr = 2;
  std::cout << "\n=== batched (ntransf=2) dim=" << dim << " ===\n";
  const int64_t n_modes[3] = {N, dim > 1 ? N : 1, dim > 2 ? N : 1};

  std::default_random_engine eng(11);
  std::uniform_real_distribution<double> dist11(-1, 1);
  auto randm11 = [&]() {
    return dist11(eng);
  };

  thrust::host_vector<double> x(M), y(M), z(M);
  thrust::host_vector<Cpx> c(2 * M);
  for (int j = 0; j < M; ++j) {
    x[j]     = PI * randm11();
    y[j]     = PI * randm11();
    z[j]     = PI * randm11();
    c[j]     = Cpx(randm11(), randm11());
    c[M + j] = 2.0 * c[j];
  }
  thrust::device_vector<double> d_x = x, d_y = y, d_z = z;
  thrust::device_vector<Cpx> d_c    = c;

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_no_cropping = 1;

  cufinufft_plan plan;
  int64_t om[3];
  if (cufinufft_makeplan(1, dim, n_modes, iflag, ntr, tol, &plan, &opts)) return 1;
  if (cufinufft_get_out_modes(plan, om)) return 1;
  int64_t ntot = 1;
  for (int d = 0; d < dim; ++d) ntot *= om[d];

  if (cufinufft_setpts(plan, M, (double *)d_x.data().get(),
                       dim > 1 ? (double *)d_y.data().get() : nullptr,
                       dim > 2 ? (double *)d_z.data().get() : nullptr, 0, nullptr,
                       nullptr, nullptr))
    return 1;
  thrust::device_vector<Cpx> d_fk(ntr * ntot);
  if (cufinufft_execute(plan, (cuDoubleComplex *)d_c.data().get(),
                        (cuDoubleComplex *)d_fk.data().get()))
    return 1;
  cufinufft_destroy(plan);
  thrust::host_vector<Cpx> fk = d_fk;

  // As above, split central from outer: the two transforms in the batch spread
  // independently, so their fw differ at roundoff, which 1/phihat amplifies in
  // the outer band. A genuine stride error would be O(1) in either band.
  double diff_c = 0, diff_o = 0, maxabs = 0;
  for (int64_t i = 0; i < ntot; ++i) maxabs = std::max(maxabs, thrust::abs(fk[i]));
  for (int64_t i = 0; i < ntot; ++i) {
    const int64_t k1 = i % om[0];
    const int64_t k2 = (i / om[0]) % om[1];
    const int64_t k3 = i / (om[0] * om[1]);
    const double d   = thrust::abs(fk[ntot + i] - 2.0 * fk[i]) / maxabs;
    double r         = std::abs(double(k1 - om[0] / 2)) / (0.5 * n_modes[0]);
    if (dim > 1) r = std::max(r, std::abs(double(k2 - om[1] / 2)) / (0.5 * n_modes[1]));
    if (dim > 2) r = std::max(r, std::abs(double(k3 - om[2] / 2)) / (0.5 * n_modes[2]));
    if (r <= 1.0)
      diff_c = std::max(diff_c, d);
    else
      diff_o = std::max(diff_o, d);
  }
  std::cout << "  extent " << om[0] << "x" << om[1] << "x" << om[2]
            << ", transform 2 vs 2*transform 1: central " << diff_c << ", outer "
            << diff_o << "\n";
  if (!(diff_c < 1e-13) || !(diff_o < 1e-5)) {
    std::cout << "  FAIL: batch stride wrong under gpu_no_cropping\n";
    return 1;
  }
  std::cout << "  batched PASS\n";
  return 0;
}

// gpu_no_cropping is type 1 only, and cannot be combined with spreadinterponly.
static int run_validation() {
  std::cout << "\n=== option validation ===\n";
  const int64_t n_modes[3] = {32, 32, 1};
  cufinufft_plan plan      = nullptr;
  int fails                = 0;

  {
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.gpu_no_cropping = 1;
    const int ier = cufinufft_makeplan(2, 2, n_modes, 1, 1, 1e-6, &plan, &opts);
    std::cout << "  type 2 + no_cropping -> ier=" << ier << "\n";
    if (ier != FINUFFT_ERR_INVALID_ARGUMENT) {
      std::cout << "  FAIL: expected FINUFFT_ERR_INVALID_ARGUMENT\n";
      ++fails;
      if (ier == 0) cufinufft_destroy(plan);
    }
  }
  {
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.gpu_no_cropping       = 1;
    opts.gpu_spreadinterponly  = 1;
    const int ier = cufinufft_makeplan(1, 2, n_modes, 1, 1, 1e-6, &plan, &opts);
    std::cout << "  spreadinterponly + no_cropping -> ier=" << ier << "\n";
    if (ier != FINUFFT_ERR_INVALID_ARGUMENT) {
      std::cout << "  FAIL: expected FINUFFT_ERR_INVALID_ARGUMENT\n";
      ++fails;
      if (ier == 0) cufinufft_destroy(plan);
    }
  }
  return fails;
}

int main() {
  std::cout << std::scientific << std::setprecision(3);
  int fails = run_validation();
  fails += run_dim(1, 256, 4000, 1e-9);
  fails += run_dim(2, 64, 4000, 1e-9);
  fails += run_dim(3, 24, 2000, 1e-9);
  fails += run_batched(2, 48, 3000, 1e-9);
  std::cout << (fails ? "\nFAILED\n" : "\nPASSED\n");
  return fails != 0;
}
