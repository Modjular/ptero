// Shared image/TIFF loader for the demo pages. Smallest possible surface:
// loadSource() and meanPlane(). See docs/GOTCHAS.md §8/§9 for the channel-
// semantics and TIFF-decoding background this is built to avoid tripping over.
//
// TIFFs are always decoded losslessly into per-plane Float32Arrays — never
// quantized to 8-bit before segmentation — regardless of whether the source
// is a multi-page channel stack, a single-page multi-sample scan, or a plain
// single-channel image. A source is "multichannel" (UI-selectable) whenever
// it genuinely has more than one plane; there is no "looks like RGB, treat it
// specially" heuristic, so a real 2- or 3-channel fluorescence scan is never
// silently misclassified as an ordinary photo.
import { decode as decodeTiff } from "https://cdn.jsdelivr.net/npm/tiff@7.1.3/+esm";

function isThumbnailSubfile(ifd) { return ((ifd.newSubfileType ?? 0) & 1) !== 0; }

function extractPlane(data, W, H, c, stride) {
  const out = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = data[i * stride + c];
  return out;
}

const COMP_NAMES = ["R", "G", "B"];

// The `tiff` package only decodes uncompressed/LZW/Deflate pixel data (Compression
// 1/5/8/32946) and only WhiteIsZero/BlackIsZero/RGB/Palette photometric interpretations
// (PhotometricInterpretation 0/1/2/3) — anything else throws a cryptic internal error
// ("Unsupported image type: N") deep in the decoder. JPEG-compressed whole-slide-image
// TIFFs (common Bio-Formats/QuPath export: Compression=7, PhotometricInterpretation=6
// YCbCr) are the case most likely to be hit here. Check the cheap tags-only decode
// (ignoreImageData skips the actual pixel-format switch that throws) so we can fail
// with an actionable message instead.
const UNSUPPORTED_COMPRESSION = { 2: "CCITT Group 3", 3: "CCITT Group 4", 6: "old-style JPEG", 7: "JPEG", 32773: "PackBits" };
const UNSUPPORTED_PHOTOMETRIC = { 4: "transparency mask", 5: "CMYK", 6: "YCbCr (often a JPEG-compressed whole-slide scan)", 8: "CIELab" };

function assertDecodable(bytes) {
  for (const ifd of decodeTiff(bytes, { ignoreImageData: true })) {
    if (isThumbnailSubfile(ifd)) continue;
    const badComp = UNSUPPORTED_COMPRESSION[ifd.compression];
    if (badComp) throw new Error(`This TIFF uses ${badComp} compression, which this in-browser decoder can't read. Re-export as uncompressed/LZW/Deflate-compressed TIFF, or convert to PNG.`);
    const badPhoto = UNSUPPORTED_PHOTOMETRIC[ifd.type];
    if (badPhoto) throw new Error(`This TIFF uses ${badPhoto} color encoding, which this in-browser decoder can't read. Re-export as an RGB/grayscale TIFF, or convert to PNG.`);
  }
}

async function decodeTiffSource(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  assertDecodable(bytes);
  let ifds = decodeTiff(bytes).filter(ifd => !isThumbnailSubfile(ifd));
  if (!ifds.length) ifds = decodeTiff(bytes);
  const { width: W, height: H } = ifds[0];
  const planes = [];
  ifds.forEach((ifd, p) => {
    if (ifd.width !== W || ifd.height !== H) return;   // skip mismatched-size pages (e.g. pyramid levels)
    const stride = ifd.components ?? 1;
    const colorSamples = stride - (ifd.alpha ? 1 : 0);   // drop a trailing alpha sample — not imaging data
    for (let c = 0; c < colorSamples; c++) {
      const label = ifds.length > 1
        ? `page ${p + 1}${colorSamples > 1 ? "." + (COMP_NAMES[c] || "c" + c) : ""}`
        : (colorSamples > 1 ? (COMP_NAMES[c] || `c${c}`) : "Gray");
      planes.push({ label, data: extractPlane(ifd.data, W, H, c, stride) });
    }
  });
  return { multi: planes.length > 1, W, H, planes };
}

async function decodeImageSource(src) {
  const done = () => { if (src.startsWith("blob:")) URL.revokeObjectURL(src); };
  const imgData = await new Promise((res, rej) => {
    const im = new Image(); im.crossOrigin = "anonymous";
    im.onload = () => {
      const c = document.createElement("canvas"); c.width = im.width; c.height = im.height;
      const ctx = c.getContext("2d"); ctx.drawImage(im, 0, 0); done();
      res(ctx.getImageData(0, 0, im.width, im.height));
    };
    im.onerror = e => { done(); rej(e); };
    im.src = src;
  });
  return { multi: false, W: imgData.width, H: imgData.height, imgData };
}

// Avoids re-decoding the same upload on every Segment click (run() and the
// preview handler both call loadSource() on the identical File object).
const tiffCache = new WeakMap();

/**
 * Loads a sample-image URL (string) or an uploaded File (TIFF or ordinary
 * image) into a uniform Source: { multi:false, W, H, imgData } for plain
 * images, or { multi, W, H, planes } for TIFFs (planes.length may be 1).
 * W/H are always present at the top level regardless of source kind.
 */
export function loadSource(input) {
  if (input instanceof File) {
    if (!/\.tiff?$/i.test(input.name)) return decodeImageSource(URL.createObjectURL(input));
    if (!tiffCache.has(input)) tiffCache.set(input, decodeTiffSource(input));
    return tiffCache.get(input);
  }
  return decodeImageSource(input);
}

export function meanPlane(planes, W, H) {
  const out = new Float32Array(W * H);
  const inv = 1 / planes.length;
  for (const p of planes) for (let i = 0; i < W * H; i++) out[i] += p.data[i] * inv;
  return out;
}
