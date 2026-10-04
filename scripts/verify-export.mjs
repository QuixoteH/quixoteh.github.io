import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const fixture = JSON.parse(
  fs.readFileSync(path.join(root, 'tests/export-contract.json'), 'utf8')
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
const renderedHtml = new Map();

for (const route of fixture.routes) {
  const file = path.join(root, 'out', route);
  if (!fs.existsSync(file)) {
    failures.push(`Missing route: ${route}`);
    continue;
  }

  const html = fs.readFileSync(file, 'utf8');
  renderedHtml.set(route, html);

  const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i);
  if (!main) {
    failures.push(`Missing <main> from ${route}`);
  } else if (decodeHtml(main[1]).length < 20) {
    failures.push(`Empty <main> content in ${route}`);
  }
}

for (const [route, page] of renderedHtml) {
  if (/style="[^"]*opacity:0/.test(page)) {
    failures.push(`Server-rendered content is hidden on ${route}`);
  }
}

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(`Verified ${fixture.routes.length} exported routes and visible page content.`);
