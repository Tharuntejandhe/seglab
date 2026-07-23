/**
 * yolo-world-detect — YOLO-World-v2 OPEN-VOCABULARY vision lane (raw onnxruntime-web).
 *
 * The arbitrary-phrase detector: unlike prompt-free YOLOE (fixed baked vocab),
 * this takes the class text features as a LIVE input (txt_feats), so any phrase
 * works. The phrase is encoded to CLIP ViT-B/32 vectors on the main side
 * (clip-text.js) and the 32 class slots are filled with the phrase + taxonomy
 * synonyms. GPU-first on every vendor (WebGPU: Metal/D3D12/Vulkan, iGPUs
 * included), fp32 there; the WASM floor runs the QDQ int8 build (~20 MB,
 * head kept fp32 — parity vs fp32: score drift ≤0.05, IoU ≥0.99). This lane
 * replaces the OWLv2 WASM floor.
 *
 * Contract (scripts/export-yolo-world.py): images[1,3,640,640] f32 +
 * txt_feats[1,32,512] f32 → output0[1,36,8400] = (cx,cy,w,h box at 640 +
 * 32 sigmoid class scores) per anchor. YOLOv8 head — NOT NMS-free, so the
 * caller thresholds, maps xywh→xyxy, and runs NMS (text-core). Boxes are
 * returned normalized [0,1] against the 640 square, matching the YOLOE lane.
 *
 * WebGPU numerical trust is earned at runtime, not assumed: the first GPU
 * result is rejected if its score field is degenerate (flat flood), and a
 * first-ever zero-det GPU result is cross-checked ONCE on wasm — disagreement
 * sticky-demotes to wasm for this worker's lifetime. Cost: at most one extra
 * wasm inference per worker spawn, only in the nothing-found case.
 *
 * onnxruntime-web is resolved vendored-first, CDN fallback (mirrors yoloe-detect).
 */

import { YOLOE_INPUT } from './text-core.js'

const SCALES = new Set(['s', 'm', 'l', 'x'])
export const YW_CLASS_SLOTS = 32 // fixed head width (scripts/export-yolo-world.py NC)

const ORT_VERSION = '1.22.0'
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`
const ORT_LOCAL = new URL('../lib/ort-web/', import.meta.url).href
const ORT_FILE = 'ort.webgpu.bundle.min.mjs'
// webgpu → fp32; wasm → QDQ int8 first (quantize_lanes.py, backbone+neck int8,
// head fp32), fp32 as the fallback when the int8 file is absent (n/x scales).
const modelURL = (scale, variant = 'fp32') => new URL(
    `../models/yolo-world/yolo-world-${scale}${variant === 'int8' ? '.int8' : ''}.onnx`, import.meta.url,
).href

let ortPromise = null
const loadOrt = () => {
    ortPromise ??= (async () => {
        for (const base of [ORT_LOCAL, ORT_CDN]) {
            try {
                const ort = await import(/* @vite-ignore */ base + ORT_FILE)
                ort.env.wasm.wasmPaths = base
                ort.env.wasm.numThreads = (typeof self !== 'undefined' && self.crossOriginIsolated)
                    ? Math.max(1, Math.min(4, ((typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4) - 1))
                    : 1
                return ort
            } catch { /* try next source */ }
        }
        throw new Error('onnxruntime-web unavailable (no vendored copy and CDN blocked)')
    })()
    return ortPromise
}

let session = null
let sessionPromise = null
let loadedScale = null
let loadedEps = null // the EP preference the session was built under
let backend = null // 'webgpu' | 'wasm'
let gpuTrusted = false // one healthy GPU result seen this worker lifetime
let gpuDemoted = false // GPU output disagreed with wasm — sticky for this worker

const buildSession = async (ort, scale, eps) => {
    let lastErr
    for (const ep of eps) {
        // Per-EP weights: the GPU gets fp32, the wasm floor gets int8-first.
        const variants = ep === 'wasm' ? ['int8', 'fp32'] : ['fp32']
        for (const variant of variants) {
            try {
                const s = await ort.InferenceSession.create(modelURL(scale, variant), {
                    executionProviders: [ep],
                    graphOptimizationLevel: 'all',
                })
                backend = ep
                return s
            } catch (err) { lastErr = err }
        }
    }
    throw lastErr || new Error('yolo-world: no execution provider available')
}

export const loadYoloWorld = (scale = 's', { webgpu = true } = {}) => {
    if (!SCALES.has(scale)) scale = 's'
    const eps = (webgpu && !gpuDemoted) ? ['webgpu', 'wasm'] : ['wasm']
    const epsKey = eps.join()
    if (session && loadedScale === scale && loadedEps === epsKey) return Promise.resolve(session)
    if (sessionPromise && loadedScale === scale && loadedEps === epsKey) return sessionPromise
    if (session) disposeYoloWorld() // scale/EP switch — release first
    loadedScale = scale
    loadedEps = epsKey
    sessionPromise = (async () => {
        const ort = await loadOrt()
        try {
            session = await buildSession(ort, scale, eps)
        } catch (err) {
            // Scale files may not be deployed ('s' is the baseline) — fall back.
            if (scale === 's') throw err
            console.warn(`[seglab] yolo-world-${scale} unavailable (${err?.message}); falling back to scale s`)
            loadedScale = 's'
            session = await buildSession(ort, 's', eps)
        }
        return session
    })()
    sessionPromise.catch(() => { if (loadedEps === epsKey) { sessionPromise = null; loadedScale = null; loadedEps = null } })
    return sessionPromise
}

let idleTimer = null
const cancelIdle = () => { if (idleTimer) { clearTimeout(idleTimer); idleTimer = null } }
const scheduleIdle = (ms) => {
    cancelIdle()
    if (!(ms > 0) || !session) return
    idleTimer = setTimeout(() => { idleTimer = null; disposeYoloWorld() }, ms)
}

export const disposeYoloWorld = () => {
    cancelIdle()
    const s = session
    session = null
    sessionPromise = null
    loadedScale = null
    loadedEps = null
    backend = null
    try { s?.release?.() } catch { /* already gone */ }
}

export const yoloWorldLoaded = () => !!session
export const yoloWorldBackend = () => backend

// A collapsed contrastive matmul floods most of the 8400 anchors past the
// threshold; healthy multi-object scenes stay well under this. (Score-spread
// alone can't be the signal here: pre-NMS duplicate anchors of one strong
// object legitimately cluster within a few hundredths.)
const FLOOD_DETS = 2000
let gpuZeroChecked = false // the once-per-lifetime zero-det wasm cross-check ran

const runInference = async (ort, s, frame, txtFeats, threshold) => {
    const side = YOLOE_INPUT // 640, shared square
    // RGB bytes → NCHW float32 [0,1].
    const d = frame.data
    const plane = side * side
    const chw = new Float32Array(3 * plane)
    for (let i = 0, p = 0; p < plane; i += 3, p += 1) {
        chw[p] = d[i] / 255
        chw[plane + p] = d[i + 1] / 255
        chw[2 * plane + p] = d[i + 2] / 255
    }
    const images = new ort.Tensor('float32', chw, [1, 3, side, side])
    const txt = new ort.Tensor('float32', txtFeats, [1, YW_CLASS_SLOTS, 512])
    const out = await s.run({ images, txt_feats: txt })
    const o0 = out[s.outputNames[0]] // [1, 4+NC, 8400]
    const [, ch, n] = o0.dims
    const data = o0.data
    const nc = ch - 4
    const dets = []
    for (let a = 0; a < n; a += 1) {
        // Best class for this anchor (scores are already sigmoid).
        let best = 0
        let cls = 0
        for (let c = 0; c < nc; c += 1) {
            const v = data[(4 + c) * n + a]
            if (v > best) { best = v; cls = c }
        }
        if (best < threshold) continue
        const cx = data[a]; const cy = data[n + a]
        const w = data[2 * n + a]; const h = data[3 * n + a]
        const x0 = (cx - w / 2) / side
        const y0 = (cy - h / 2) / side
        const x1 = (cx + w / 2) / side
        const y1 = (cy + h / 2) / side
        dets.push({ box: [x0, y0, x1, y1], score: best, classIdx: cls })
    }
    return dets
}

/**
 * Detect over `frame` — { data: RGB bytes, width, height } letterboxed into the
 * 640² square (top-left) by the caller — conditioned on `txtFeats`: a Float32Array
 * of YW_CLASS_SLOTS×512 L2-normalized CLIP vectors (clip-text.js). Returns
 * { dets: [{ box:[x0,y0,x1,y1] normalized [0,1] to the square, score, classIdx }],
 *   backend }. classIdx indexes the slot list the caller supplied. No NMS here.
 * `webgpu:false` (policy/pressure) pins the wasm floor for this call's session.
 */
export const detectYoloWorld = async ({ frame, txtFeats, threshold = 0.25, scale = 's', webgpu = true, dispose = false, idleMs = 0 }) => {
    cancelIdle()
    const ort = await loadOrt()
    const s = await loadYoloWorld(scale, { webgpu })
    try {
        let dets = await runInference(ort, s, frame, txtFeats, threshold)
        // Earn GPU trust at runtime: reject a flooded score field outright, and
        // cross-check the first-ever empty GPU result on wasm. Either mismatch
        // sticky-demotes this worker to wasm; agreement trusts the GPU for good.
        if (backend === 'webgpu' && !gpuTrusted) {
            const flooded = dets.length >= FLOOD_DETS
            if (flooded || (dets.length === 0 && !gpuZeroChecked)) {
                if (dets.length === 0) gpuZeroChecked = true
                disposeYoloWorld()
                const w = await loadYoloWorld(scale, { webgpu: false })
                const wasmDets = await runInference(ort, w, frame, txtFeats, threshold)
                if (flooded || wasmDets.length > 0) {
                    gpuDemoted = true
                    dets = wasmDets
                } else {
                    gpuTrusted = true // both agree the scene is a miss
                    dets = wasmDets
                }
            } else if (dets.length > 0) {
                gpuTrusted = true
            }
        }
        return { dets, backend }
    } finally {
        if (dispose) disposeYoloWorld()
        else scheduleIdle(idleMs)
    }
}
