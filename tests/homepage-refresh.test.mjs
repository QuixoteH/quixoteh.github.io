import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { parse } from 'smol-toml';

const root = path.resolve(import.meta.dirname, '..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const readToml = (file) => parse(read(file));
const isNonEmptyString = (value) => typeof value === 'string' && value.trim().length > 0;

test('editable content files are structurally valid', () => {
  const config = readToml('content/config.toml');
  for (const value of [config.site?.title, config.author?.name, config.author?.title, config.social?.email]) {
    assert.ok(isNonEmptyString(value));
  }

  assert.ok(Array.isArray(config.navigation));
  for (const item of config.navigation) {
    assert.ok(isNonEmptyString(item.title));
    assert.ok(isNonEmptyString(item.href));
  }

  const about = readToml('content/about.toml');
  assert.ok(Array.isArray(about.sections));
  for (const section of about.sections) {
    assert.ok(isNonEmptyString(section.id));
    assert.ok(isNonEmptyString(section.type));
    assert.ok(isNonEmptyString(section.title));
    if (section.source) {
      assert.ok(fs.existsSync(path.join(root, 'content', section.source)));
    }
  }

  for (const file of ['education.toml', 'experience.toml', 'portfolio.toml', 'teaching.toml']) {
    const page = readToml(`content/${file}`);
    assert.equal(page.type, 'card');
    assert.ok(isNonEmptyString(page.title));
    assert.ok(Array.isArray(page.items));
    for (const item of page.items) {
      assert.ok(isNonEmptyString(item.title));
      for (const field of ['subtitle', 'date', 'content']) {
        if (item[field] !== undefined) {
          assert.ok(isNonEmptyString(item[field]));
        }
      }
      if (item.image?.startsWith('/')) {
        assert.ok(fs.existsSync(path.join(root, 'public', item.image.slice(1))));
      }
      if (item.tags !== undefined) {
        assert.ok(Array.isArray(item.tags));
        assert.ok(item.tags.every(isNonEmptyString));
      }
    }
  }

  const news = readToml('content/news.toml');
  assert.ok(Array.isArray(news.news));
  for (const item of news.news) {
    assert.ok(isNonEmptyString(item.date));
    assert.ok(isNonEmptyString(item.content));
  }

  for (const file of ['bio.md', 'cv-json.md', 'publications.bib']) {
    assert.ok(isNonEmptyString(read(`content/${file}`)));
  }
});

test('profile assets and homepage renderers stay wired', () => {
  const profile = read('src/components/home/Profile.tsx');
  assert.match(profile, /Github, Linkedin, Mail, MapPin/);
  assert.match(profile, /name: 'Email'/);
  assert.match(profile, /`mailto:\$\{social\.email\}`/);

  const page = read('src/app/page.tsx');
  const client = read('src/components/home/HomePageClient.tsx');
  assert.match(page, /type: 'markdown' \| 'publications' \| 'list' \| 'card'/);
  assert.match(page, /case 'card'/);
  assert.match(client, /case 'card'/);
  assert.doesNotMatch(client, /SelectedPublications/);

  const favicon = read('public/favicon-book.svg');
  assert.match(favicon, /<title>Open book<\/title>/);
});

test('GentleFress visual structure and light-default theme stay configured', () => {
  const card = read('src/components/pages/CardPage.tsx');
  const news = read('src/components/home/News.tsx');
  const store = read('src/lib/stores/themeStore.ts');
  const layout = read('src/app/layout.tsx');
  const css = read('src/app/globals.css');
  const textPage = read('src/components/pages/TextPage.tsx');

  assert.match(card, /item\.image/);
  assert.match(card, /h-12 w-12/);
  assert.match(news, /max-h-80/);
  assert.match(news, /overflow-y-auto/);
  assert.match(news, /ReactMarkdown/);
  assert.match(store, /theme: 'light'/);
  assert.match(layout, /parsed\?\.state\?\.theme \|\| 'light'/);
  assert.match(css, /--accent: #7D6B8C;/);
  assert.match(css, /--accent-dark: #675873;/);
  assert.match(textPage, /h3: \(\{ children \}\) => <h3 className="text-xl font-serif font-semibold/);
});

test('export verification checks structure instead of frozen content', () => {
  const contract = JSON.parse(read('tests/export-contract.json'));
  assert.deepEqual(Object.keys(contract), ['routes']);
  assert.ok(Array.isArray(contract.routes));
  assert.equal(new Set(contract.routes).size, contract.routes.length);
  assert.ok(contract.routes.every(isNonEmptyString));

  const verifier = read('scripts/verify-export.mjs');
  assert.match(verifier, /<main\\b/);
  assert.match(verifier, /opacity:0/);
  assert.doesNotMatch(verifier, /fixture\.(required|requiredHtml|forbiddenByRoute|banned)/);
});
