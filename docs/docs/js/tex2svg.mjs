// Bake LaTeX into an SVG diagram so it renders as real math inside a plain <img>.
// Two tags in the source SVG, both replaced by MathJax-rendered SVG (glyphs as paths):
//   <tex x="415" y="514">\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)</tex>      pure math
//   <txt x="40" y="96" bold="1">Actor learns $H$, i.e. $\pi$ · 行动者</txt>       text with $inline math$
// Attributes:
//   x, y    position; y is the text baseline (same as <text>)
//   size    font size in px (default 13), fill colour (default #3d3d3a)
//   anchor  start | middle | end, like text-anchor (default: middle for <tex>, start for <txt>)
//   bold    any value makes <txt> text bold
// Latin text and math use the TeX (Computer Modern) font; CJK characters are not in it, so MathJax
// emits them as <text> elements, styled by the `text{...}` rule in the source SVG
// (set font-size:1000px there: MathJax reserves 1em per CJK char but draws it at 0.884em).
// Each label also gets an invisible <text> holding its source, so it can be selected and copied
// when the SVG is opened directly or embedded with <object> (never inside an <img>).
// Usage (from docs/):  node docs/js/tex2svg.mjs in.src.svg out.svg
import fs from 'fs';
import { mathjax } from 'mathjax-full/js/mathjax.js';
import { TeX } from 'mathjax-full/js/input/tex.js';
import 'mathjax-full/js/input/tex/base/BaseConfiguration.js';
import 'mathjax-full/js/input/tex/ams/AmsConfiguration.js';
import { SVG } from 'mathjax-full/js/output/svg.js';
import { liteAdaptor } from 'mathjax-full/js/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from 'mathjax-full/js/handlers/html.js';

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', {
  InputJax: new TeX({ packages: ['base', 'ams'] }),
  OutputJax: new SVG({ fontCache: 'global' }), // each glyph defined once for the whole file, then <use>d
});

const attr = (s, name, dflt) => (s.match(new RegExp(`\\b${name}="([^"]*)"`)) || [])[1] ?? dflt;
const ex = (s) => parseFloat(s); // MathJax sizes come as "12.3ex"
const unescapeXml = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');

// "Actor learns $H$" -> "\text{Actor learns }H"  (\text{} is literal in MathJax, so no escaping)
function textToTex(content, bold) {
  const wrap = bold ? '\\textbf' : '\\text';
  return unescapeXml(content.trim())
    .split('$')
    .map((part, i) => (i % 2 ? part : part && `${wrap}{${part}}`))
    .join('');
}

const escapeXml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// `copy` is the label's source text: laid over the paths as invisible real <text>, so it can be selected and copied
function render(attrs, tex, defaultAnchor, copy) {
  const x = +attr(attrs, 'x'), y = +attr(attrs, 'y');
  const size = +attr(attrs, 'size', 13);
  const fill = attr(attrs, 'fill', '#3d3d3a');
  const anchor = attr(attrs, 'anchor', defaultAnchor);

  const svg = adaptor.firstChild(doc.convert(tex, { display: false }));
  const px = 0.442 * size; // 1ex in px at this font size (MathJax TeX font)
  const w = ex(adaptor.getAttribute(svg, 'width')) * px;
  const h = ex(adaptor.getAttribute(svg, 'height')) * px;
  const valign = (adaptor.getAttribute(svg, 'style') || '').match(/vertical-align:\s*([-\d.]+)ex/);
  const depth = valign ? -ex(valign[1]) * px : 0; // no vertical-align when nothing goes below the baseline
  const left = { start: x, middle: x - w / 2, end: x - w }[anchor];

  adaptor.setAttribute(svg, 'x', left.toFixed(2));
  adaptor.setAttribute(svg, 'y', (y - h + depth).toFixed(2));
  adaptor.setAttribute(svg, 'width', w.toFixed(2));
  adaptor.setAttribute(svg, 'height', h.toFixed(2));
  adaptor.setAttribute(svg, 'color', fill); // MathJax paints with currentColor
  adaptor.setAttribute(svg, 'overflow', 'visible'); // CJK <text> glyphs may poke past the estimated box
  adaptor.setAttribute(svg, 'style', 'user-select:none'); // CJK <text> inside would otherwise be copied twice
  adaptor.removeAttribute(svg, 'role');
  adaptor.removeAttribute(svg, 'focusable');
  // inline style, not a font-size attribute: the source's text{font-size:1000px} rule would override that
  const overlay = `<text x="${left.toFixed(2)}" y="${y}" style="font-size:${size}px" fill="transparent"`
    + ` textLength="${w.toFixed(2)}" lengthAdjust="spacingAndGlyphs">${escapeXml(copy)}</text>`;
  return adaptor.outerHTML(svg) + overlay;
}

const [src, out] = process.argv.slice(2);
const result = fs
  .readFileSync(src, 'utf8')
  .replace(/<tex\b([^>]*)>([\s\S]*?)<\/tex>/g, (_, attrs, tex) =>
    render(attrs, unescapeXml(tex.trim()), 'middle', `$${unescapeXml(tex.trim())}$`))
  .replace(/<txt\b([^>]*)>([\s\S]*?)<\/txt>/g, (_, attrs, content) =>
    render(attrs, textToTex(content, attr(attrs, 'bold') !== undefined), 'start', unescapeXml(content.trim())));
// glyph shapes shared by all labels, written once right before </svg>
const defs = adaptor.outerHTML(doc.outputJax.fontCache.getCache());
fs.writeFileSync(out, result.replace(/<\/svg>\s*$/, `${defs}\n</svg>\n`));
