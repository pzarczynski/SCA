use pyo3::{BoundObject, prelude::*};
use rayon::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::mem::MaybeUninit;

#[derive(PartialEq, Clone, Copy)]
struct Score(f32, usize, usize);

impl Eq for Score {}

impl PartialOrd for Score {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl Ord for Score {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn alloc_uninit_vec(size: usize) -> Vec<MaybeUninit<f32>> {
    let mut c: Vec<MaybeUninit<f32>> = Vec::with_capacity(size);
    unsafe { c.set_len(size); }
    c
}

fn assume_init(mut v: Vec<MaybeUninit<f32>>) -> Vec<f32> {
    let size = v.len();
    let ptr = v.as_mut_ptr() as *mut f32;
    std::mem::forget(v);
    unsafe { Vec::from_raw_parts(ptr, size, size) }
}

#[inline(always)]
fn fused_oneway_fstat(
    xi: &[f32],
    xj: &[f32],
    off: &[usize],
    n_classes: usize,
    n_total: u32,
) -> f32 {
    let mut ss_btw = 0.0f32;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;

    for c in 0..n_classes {
        let start = off[c];
        let end = off[c + 1];
        let n_g = (end - start) as u32;

        if n_g == 0 { continue; }

        let (sum_g, sum_sq_g) = xi[start..end]
            .iter()
            .zip(&xj[start..end])
            .fold((0.0, 0.0), |(sum, sum_sq), (&a, &b)| {
                let val = a * b; (sum + val, sum_sq + val * val)
            });

        ss_btw += (sum_g * sum_g) / (n_g as f32);
        sum += sum_g;
        sum_sq += sum_sq_g;
    }

    let cor = (sum * sum) / (n_total as f32);
    ss_btw -= cor;
    
    let ss_tot = sum_sq - cor;
    let ss_wth = ss_tot - ss_btw;

    let ms_btw = ss_btw / (n_classes - 1) as f32;
    let ms_wth = ss_wth / (n_total - n_classes as u32) as f32;

    if ms_wth <= f32::EPSILON { return 0.0; }
    ms_btw / ms_wth
}

fn compute_class_offsets(
    y: ndarray::ArrayView1<'_, usize>, 
    n_cls: usize
) -> Vec<usize> {
    let mut counts = vec![0; n_cls];
    for &yi in y.iter() { counts[yi] += 1; }

    let mut off = vec![0; n_cls + 1];
    for c in 0..n_cls { off[c + 1] = off[c] + counts[c]; }
    off
}

fn transpose_and_group(
    x: ndarray::ArrayView2<'_, f32>,
    y: ndarray::ArrayView1<'_, usize>,
    off: &[usize],
    means: &[f32],
    scales: &[f32],
) -> Vec<f32> {
    let n_samp = x.nrows();
    let n_feats = x.ncols();
    let size = n_feats * n_samp;

    let mut c = alloc_uninit_vec(size);

    c.par_chunks_mut(n_samp).enumerate()
        .for_each(|(j, col_out)| {
            let mut cur_off = off.to_vec();
            for (i, &yi) in y.iter().enumerate() {
                col_out[cur_off[yi]].write((x[[i, j]] - means[j]) / scales[j]);
                cur_off[yi] += 1;
            }
        });

    assume_init(c)
}

fn find_top_polynomial_pairs(
    grouped_cols: &[f32],
    off: &[usize],
    n_cls: usize,
    n_samp: usize,
    n_feat: usize,
    m: usize,
    k: isize,
) -> Vec<(f32, usize, usize)> {
    let heap = (0..n_feat).into_par_iter()
        .fold(
            || BinaryHeap::with_capacity(m + 1),
            |mut heap, i| {
                let xi = &grouped_cols[i * n_samp .. (i + 1) * n_samp];
                let end = (i as isize + k).max(0) as usize;

                for j in 0..end.min(n_feat) {
                    let xj = &grouped_cols[j * n_samp .. (j + 1) * n_samp];
                    
                    let fstat = fused_oneway_fstat(xi, xj, off, n_cls, n_samp as u32);
                    if fstat.is_nan() { continue; }

                    heap.push(Score(fstat, i, j));
                    if heap.len() > m { heap.pop(); }
                }
                heap
            },
        )
        .reduce(
            || BinaryHeap::with_capacity(m),
            |mut h1, mut h2| {
                for item in h2.drain() {
                    h1.push(item);
                    if h1.len() > m { h1.pop(); }
                }
                h1
            },
        );

    let mut results: Vec<(f32, usize, usize)> = heap
        .into_iter()
        .map(|Score(f, i, j)| (f, i, j))
        .collect();

    results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    results
}

fn compute_means_and_scales(
    x: ndarray::ArrayView2<'_, f32>
) -> (Vec<f32>, Vec<f32>) {
    let n_samp = x.nrows() as f32;
    let mut means = alloc_uninit_vec(x.ncols());
    let mut scales = alloc_uninit_vec(x.ncols());

    means.par_iter_mut()
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(j, (mean, scale))| {
            let xj = x.column(j);
            let m = xj.fold(0.0, |acc, &val| acc + val) / n_samp;
            mean.write(m);

            let var = xj.fold(0.0, |a, &v| { a + (v - m).powi(2) }) / n_samp;
            scale.write(if var < f32::EPSILON { 1.0 } else { var.sqrt() });
        });

    (assume_init(means), assume_init(scales))
}

#[pyfunction]
fn fused_select_poly<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f32>,
    y: PyReadonlyArray1<'_, usize>,
    n_cls: usize,
    m: usize,
    k: isize,
) -> PyResult<(Vec<(f32, usize, usize)>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let x = x.as_array();
    let y = y.as_array();
    
    let (means, scales) = compute_means_and_scales(x);
    let off = compute_class_offsets(y, n_cls);
    let grouped_cols = transpose_and_group(x, y, &off, &means, &scales);
    
    let results = find_top_polynomial_pairs(
        &grouped_cols, &off, n_cls, x.nrows(), x.ncols(), m, k,
    );

    let means_py = PyArray1::from_vec(py, means);
    let scales_py = PyArray1::from_vec(py, scales);
    Ok((results, means_py, scales_py))
}

#[pyfunction]
fn fused_transform_poly<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f32>,
    means: PyReadonlyArray1<'_, f32>,
    scales: PyReadonlyArray1<'_, f32>,
    ix: PyReadonlyArray1<'_, usize>,
    jx: PyReadonlyArray1<'_, usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x = x.as_array();
    let m = means.as_array().to_vec();
    let s = scales.as_array().to_vec();
    let ix = ix.as_array().to_vec();
    let jx = jx.as_array().to_vec();

    let out = unsafe {
        PyArray2::<f32>::new(py, [x.nrows(), ix.len()], false).into_bound() 
    };
    let mut out_arr = unsafe { out.as_array_mut() };

    ndarray::Zip::from(out_arr.rows_mut())
        .and(x.rows())
        .par_for_each(|mut out_row, x_row| {
            for c in 0..ix.len() {
                let i = ix[c];
                let j = jx[c];
                
                let val_i = (x_row[i] - m[i]) / s[i];
                let val_j = (x_row[j] - m[j]) / s[j];

                out_row[c] = val_i * val_j;
            }
        });

    Ok(out)
}

#[pymodule]
fn sca_fused_ops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused_select_poly, m)?)?;
    m.add_function(wrap_pyfunction!(fused_transform_poly, m)?)?;
    Ok(())
}