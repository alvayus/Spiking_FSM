/* Transition conditions: Boolean expressions over the inputs, written with
 * ¬ (NOT), ∧ (AND) and ∨ (OR). They are parsed, converted to a sum of products
 * (the "terms" stored in the machine, one AND gate each) and printed back. */
"use strict";

// Boolean operators used to write and display transition conditions.
const OP = { not: "¬", and: "∧", or: "∨" };
function literals(opt) {
  return spec.inputs.filter((i) => opt[i]).map((i) => (opt[i] === "on" ? i : OP.not + i));
}
function optionText(opt) { return literals(opt).join(` ${OP.and} `) || "any input spike"; }
// Condition as typed by the user (t.expr) or, failing that, as a sum of products.
function transLabel(t) {
  if (t.expr) return t.expr;
  if (t.options.length === 1) return optionText(t.options[0]);
  return t.options.map((o) => (literals(o).length > 1 ? `(${optionText(o)})` : optionText(o))).join(` ${OP.or} `);
}

// ---- expression parser:  expr := term (∨ term)* ; term := factor (∧ factor)* ;
//      factor := ¬ factor | ( expr ) | INPUT, with optional postfix ' (NOT)
function tokenize(text) {
  const toks = [], re = /\s*(?:(\|\|?|∨|\+|\bOR\b)|(&&?|∧|·|\*|\bAND\b)|(¬|!|~|\bNOT\b)|(\()|(\))|(')|([A-Za-z_][A-Za-z0-9_]*))/iy;
  let pos = 0;
  text = text.trimEnd();
  while (pos < text.length) {
    re.lastIndex = pos;
    const m = re.exec(text);
    if (!m) throw new Error(`Unexpected character "${text.slice(pos).trim()[0]}"`);
    pos = re.lastIndex;
    if (m[1]) toks.push({ k: "or" }); else if (m[2]) toks.push({ k: "and" }); else if (m[3]) toks.push({ k: "not" });
    else if (m[4]) toks.push({ k: "(" }); else if (m[5]) toks.push({ k: ")" }); else if (m[6]) toks.push({ k: "'" });
    else toks.push({ k: "id", v: m[7] });
  }
  return toks;
}
function parseExpr(text) {
  const toks = tokenize(text); let i = 0;
  const peek = () => toks[i] && toks[i].k;
  const expr = () => { const xs = [term()]; while (peek() === "or") { i++; xs.push(term()); } return xs.length > 1 ? { op: "or", xs } : xs[0]; };
  const term = () => { const xs = [factor()]; while (peek() === "and") { i++; xs.push(factor()); } return xs.length > 1 ? { op: "and", xs } : xs[0]; };
  const factor = () => {
    let node;
    const t = toks[i++];
    if (!t) throw new Error("The expression ends too early");
    if (t.k === "not") node = { op: "not", x: factor() };
    else if (t.k === "(") { node = expr(); if (peek() !== ")") throw new Error("A parenthesis is not closed"); i++; }
    else if (t.k === "id") {
      if (!spec.inputs.includes(t.v)) throw new Error(`"${t.v}" is not an input of the machine (${spec.inputs.join(", ")})`);
      node = { op: "id", v: t.v };
    } else throw new Error(t.k === ")" ? "Unexpected closing parenthesis" : "An operator is missing an operand");
    while (peek() === "'") { i++; node = { op: "not", x: node }; }
    return node;
  };
  if (!toks.length) throw new Error("Write a condition, e.g. A ∧ ¬B");
  const ast = expr();
  if (i < toks.length) throw new Error(toks[i].k === "id" || toks[i].k === "(" ? `Missing operator before "${toks[i].v || "("}"` : "Unexpected closing parenthesis");
  return ast;
}
function prettyExpr(n, parent = 0) {   // precedence: or 1, and 2, not 3
  if (n.op === "id") return n.v;
  if (n.op === "not") return OP.not + prettyExpr(n.x, 3);
  const prec = n.op === "or" ? 1 : 2;
  const s = n.xs.map((x) => prettyExpr(x, prec)).join(` ${OP[n.op]} `);
  return prec <= parent ? `(${s})` : s;   // also bracket ∧ inside ∨, for readability
}
// Disjunctive normal form: list of terms {input: "on"|"off"}; [] = never, [{}] = always.
function toDnf(n) {
  const and = (A, B) => {
    const out = [];
    for (const a of A) for (const b of B) {
      const m = { ...a }; let ok = true;
      for (const [k, v] of Object.entries(b)) { if (m[k] && m[k] !== v) { ok = false; break; } m[k] = v; }
      if (ok) out.push(m);
    }
    return out;
  };
  if (n.op === "id") return [{ [n.v]: "on" }];
  if (n.op === "or") return n.xs.flatMap(toDnf);
  if (n.op === "and") return n.xs.map(toDnf).reduce(and);
  // NOT: De Morgan over the DNF of the operand
  return toDnf(n.x).map((term) => Object.entries(term).map(([k, v]) => ({ [k]: v === "on" ? "off" : "on" })))
    .reduce(and, [{}]);
}
// Remove duplicate and absorbed terms (A ∨ (A ∧ B) = A), keep the inputs in machine order.
function simplifyDnf(terms) {
  const sub = (a, b) => Object.entries(a).every(([k, v]) => b[k] === v);   // literals of a ⊆ literals of b
  const keep = terms.filter((t, k) => !terms.some((u, j) => j !== k && sub(u, t) && (!sub(t, u) || j < k)));
  return keep.map((t) => Object.fromEntries(spec.inputs.filter((i) => t[i]).map((i) => [i, t[i]])));
}
