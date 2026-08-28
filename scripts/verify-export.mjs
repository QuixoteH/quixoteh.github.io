import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const fixture = JSON.parse(
  fs.readFileSync(path.join(root, 'tests/baseline-content.json'), 'utf8')
);

const decodeHtml = (value) => value
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
  .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
  .replace(/<[^>]+>/g, ' ')
  .replace(/&quot;/g, '"')
  .replace(/&#x27;|&#39;/g, "'")
  .replace(/&amp;/g, '&')
  .replace(/&ndash;/g, '–')
  .replace(/&mdash;/g, '—')
  .replace(/&lt;/g, '<')
  .replace(/&gt;/g, '>')
  .replace(/\s+/g, ' ')
  .replace(/\s+([,.;:])/g, '$1')
  .trim();

const failures = [];
const renderedText = new Map();
const renderedHtml = new Map();

for (const route of fixture.routes) {
  const file = path.join(root, 'out', route);
  if (!fs.existsSync(file)) {
    failures.push(`Missing route: ${route}`);
    continue;
  }

  const html = fs.readFileSync(file, 'utf8');
  renderedHtml.set(route, html);
  renderedText.set(route, decodeHtml(html));
}

for (const [route, required] of Object.entries(fixture.required)) {
  const page = renderedText.get(route) || '';
  for (const text of required) {
    if (!page.includes(text)) {
      failures.push(`Missing from ${route}: ${text}`);
    }
  }
}

for (const [route, required] of Object.entries(fixture.requiredHtml)) {
  const page = renderedHtml.get(route) || '';
  for (const text of required) {
    if (!page.includes(text)) {
      failures.push(`Missing HTML from ${route}: ${text}`);
    }
  }
}

for (const [route, page] of renderedHtml) {
  if (/style="[^"]*opacity:0/.test(page)) {
    failures.push(`Server-rendered content is hidden on ${route}`);
  }
}

const corpus = [...renderedText.values()].join(' ');
for (const text of fixture.banned) {
  if (corpus.includes(text)) {
    failures.push(`Example content remains: ${text}`);
  }
}

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(`Verified ${fixture.routes.length} routes and preserved content.`);
