# GentleFress Homepage Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Zhe Li's GentleFress homepage structure on Hai Huang's existing PRISM site, default new visitors to light mode, use a Morandi-purple accent, and correct only the confirmed stale CV facts.

**Architecture:** Extend the existing homepage section loader with PRISM's card-section behavior, keep all editable facts in TOML/Markdown, and reuse the shared `CardPage` and `News` renderers. Keep standalone routes unchanged, add source/content contract tests before each implementation slice, then verify the static export and the deployed GitHub Pages site.

**Tech Stack:** Next.js 15 static export, React 19, TypeScript, Tailwind CSS 4, TOML/Markdown content, Node.js 22 test runner, GitHub Actions Pages deployment.

## Global Constraints

- Baseline is `master` commit `e1501b3`; the approved design commit is `49d8f3a`.
- Navigation order is exactly `About / Research / Publications / Awards / CV`.
- Homepage order is exactly `About / Education / Experience / News`.
- Experience entries are exactly `A*STAR Research Attachment` and `China Unicom AI Solutions Engineer Intern`, each with only a `title` field.
- Do not add institution, date, description, tags, dissertation wording, or SIMTech wording to Experience.
- Keep the current biography's visible wording and all seven News entries unchanged.
- Keep CV degree abbreviations `M.S.` and `B.E.` unchanged.
- Keep the MSM-Seg CV/publication status unchanged.
- Remove `Selected Publications` from the homepage only.
- First visit defaults to light; explicit stored theme choices remain supported.
- Light-theme accent is exactly `#7D6B8C`.
- Do not change Research or Awards content or layout.
- Do not introduce any GentleFress personal text, links, emoji, or example content.
- Preserve SSR visibility: no main exported route may contain server-rendered `opacity:0`.
- Use the current primary model inline; do not delegate to Luna or subagents.

---

### Task 1: Homepage Content Contract And Card Sections

**Files:**
- Create: `tests/homepage-refresh.test.mjs`
- Create: `content/education.toml`
- Create: `content/experience.toml`
- Create: `public/logos/ntu.svg`
- Create: `public/logos/neau.png`
- Modify: `package.json`
- Modify: `content/config.toml`
- Modify: `content/about.toml`
- Modify: `content/bio.md`
- Modify: `src/app/page.tsx`
- Modify: `src/components/home/HomePageClient.tsx`

**Interfaces:**
- Consumes: existing `getTomlContent<T>()`, `CardPageConfig`, `CardPage`, `About`, and `News`.
- Produces: homepage `SectionConfig` support for `{ type: 'card'; source: string; config?: CardPageConfig }` and two TOML-backed card sections.

- [ ] **Step 1: Add the Node test script and write the failing homepage-content contract**

Add `"test": "node --test tests/*.test.mjs"` to `package.json` and prepend it to `check`:

```json
{
  "scripts": {
    "dev": "next dev --turbopack",
    "build": "next build && node scripts/create-redirects.mjs",
    "start": "next start",
    "lint": "eslint .",
    "test": "node --test tests/*.test.mjs",
    "verify:export": "node scripts/verify-export.mjs",
    "check": "npm test && npm run build && npm run verify:export"
  }
}
```

Create `tests/homepage-refresh.test.mjs`:

```js
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { parse } from 'smol-toml';

const root = path.resolve(import.meta.dirname, '..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const readToml = (file) => parse(read(file));

test('homepage config matches the approved GentleFress structure', () => {
  const config = readToml('content/config.toml');
  assert.equal(config.author.title, 'M.Sc. Student in Robotics and Intelligent Systems');
  assert.deepEqual(
    config.navigation.map(({ title, href }) => ({ title, href })),
    [
      { title: 'About', href: '/' },
      { title: 'Research', href: '/portfolio/' },
      { title: 'Publications', href: '/publications/' },
      { title: 'Awards', href: '/teaching/' },
      { title: 'CV', href: '/cv-json/' },
    ]
  );

  const about = readToml('content/about.toml');
  assert.equal(about.title, 'About');
  assert.deepEqual(
    about.sections.map(({ id, type, source, title }) => ({ id, type, source, title })),
    [
      { id: 'about', type: 'markdown', source: 'bio.md', title: 'About' },
      { id: 'education', type: 'card', source: 'education.toml', title: 'Education' },
      { id: 'experience', type: 'card', source: 'experience.toml', title: 'Experience' },
      { id: 'news', type: 'list', source: 'news.toml', title: 'News' },
    ]
  );
});

test('education and experience contain only approved facts', () => {
  const education = readToml('content/education.toml');
  assert.deepEqual(education.items, [
    {
      title: 'M.S. in Robotics and Intelligent Systems',
      subtitle: 'Nanyang Technological University',
      date: 'Aug. 2026 - Jan. 2028 (Expected)',
      content: 'School of Mechanical and Aerospace Engineering',
      image: '/logos/ntu.svg',
    },
    {
      title: 'B.E. in Internet of Things',
      subtitle: 'Northeast Agricultural University',
      date: 'Sep. 2022 - Jun. 2026',
      content: 'College of Intelligent Science and Engineering\n\nGPA: 85/100',
      image: '/logos/neau.png',
    },
  ]);

  const experience = readToml('content/experience.toml');
  assert.deepEqual(experience.items, [
    { title: 'A*STAR Research Attachment' },
    { title: 'China Unicom AI Solutions Engineer Intern' },
  ]);
  for (const item of experience.items) {
    assert.deepEqual(Object.keys(item), ['title']);
  }
});

test('homepage loader and renderer support card sections', () => {
  const page = read('src/app/page.tsx');
  const client = read('src/components/home/HomePageClient.tsx');
  assert.match(page, /type: 'markdown' \| 'publications' \| 'list' \| 'card'/);
  assert.match(page, /case 'card'/);
  assert.match(client, /case 'card'/);
  assert.doesNotMatch(client, /SelectedPublications/);
});
```

- [ ] **Step 2: Run the new tests and verify the expected failure**

Run:

```bash
npm test
```

Expected: FAIL because `author.title` is empty, `About` is absent from navigation, the two new TOML files do not exist, and card sections are unsupported.

- [ ] **Step 3: Add card-section support to the server loader**

In `src/app/page.tsx`, extend `SectionConfig`:

```ts
interface SectionConfig {
  id: string;
  type: 'markdown' | 'publications' | 'list' | 'card';
  title?: string;
  source?: string;
  filter?: string;
  limit?: number;
  content?: string;
  publications?: Publication[];
  items?: NewsItem[];
  config?: CardPageConfig;
}
```

Add this branch to `processSections` after `case 'list'`:

```ts
      case 'card': {
        const config = section.source
          ? getTomlContent<CardPageConfig>(section.source, locale)
          : null;
        return {
          ...section,
          config: config || undefined,
        };
      }
```

- [ ] **Step 4: Render card sections and remove the homepage publication renderer**

In `src/components/home/HomePageClient.tsx`, remove the `SelectedPublications` import, extend its local `SectionConfig` exactly as in Step 3, and replace the publications branch with the card branch while keeping Markdown and News:

```tsx
                  case 'card':
                    return section.config ? (
                      <CardPage
                        key={section.id}
                        config={section.config}
                        embedded={true}
                      />
                    ) : null;
```

Do not delete `src/components/home/SelectedPublications.tsx`; it remains part of the PRISM template but is no longer imported by the homepage.

- [ ] **Step 5: Configure navigation, profile title, and homepage sections**

Set `content/config.toml` author title and navigation to:

```toml
[author]
name = "Hai HUANG"
title = "M.Sc. Student in Robotics and Intelligent Systems"
institution = "Nanyang Technological University"
avatar = "/profile.png"

[[navigation]]
title = "About"
type = "page"
target = "about"
href = "/"

[[navigation]]
title = "Research"
type = "page"
target = "portfolio"
href = "/portfolio/"

[[navigation]]
title = "Publications"
type = "page"
target = "publications"
href = "/publications/"

[[navigation]]
title = "Awards"
type = "page"
target = "teaching"
href = "/teaching/"

[[navigation]]
title = "CV"
type = "page"
target = "cv-json"
href = "/cv-json/"
```

Replace the heading and sections in `content/about.toml` with:

```toml
type = "about"
title = "About"

[profile]
research_interests = [
  "Force-aware Robot Learning",
  "Robot Manipulation",
  "Vision-Language-Action Models"
]

[[sections]]
id = "about"
type = "markdown"
source = "bio.md"
title = "About"

[[sections]]
id = "education"
type = "card"
source = "education.toml"
title = "Education"

[[sections]]
id = "experience"
type = "card"
source = "experience.toml"
title = "Experience"

[[sections]]
id = "news"
type = "list"
title = "News"
source = "news.toml"
```

- [ ] **Step 6: Add Education and exact single-line Experience data**

Create `content/education.toml`:

```toml
type = "card"
title = "Education"

[[items]]
title = "M.S. in Robotics and Intelligent Systems"
subtitle = "Nanyang Technological University"
date = "Aug. 2026 - Jan. 2028 (Expected)"
content = "School of Mechanical and Aerospace Engineering"
image = "/logos/ntu.svg"

[[items]]
title = "B.E. in Internet of Things"
subtitle = "Northeast Agricultural University"
date = "Sep. 2022 - Jun. 2026"
content = "College of Intelligent Science and Engineering\n\nGPA: 85/100"
image = "/logos/neau.png"
```

Create `content/experience.toml`:

```toml
type = "card"
title = "Experience"

[[items]]
title = "A*STAR Research Attachment"

[[items]]
title = "China Unicom AI Solutions Engineer Intern"
```

- [ ] **Step 7: Add links without changing the biography's visible wording**

Replace `content/bio.md` with this single paragraph:

```markdown
Hai Huang (Student Member, IEEE) received the **B.Eng. degree in Internet of Things Engineering** from [Northeast Agricultural University](https://english.neau.edu.cn/), Harbin, China, in 2026. He is currently pursuing the **M.Sc. degree in Robotics and Intelligent Systems** at [Nanyang Technological University](https://www.ntu.edu.sg/), Singapore. He is conducting dissertation research with the [Singapore Institute of Manufacturing Technology (SIMTech)](https://www.a-star.edu.sg/simtech), [Agency for Science, Technology and Research (A\*STAR)](https://www.a-star.edu.sg/). Previously, he was an **AI Solutions Engineer Intern** at the Digital Technology Center, China United Network Communications Group Co., Ltd. His research interests focus on **robot learning and manipulation**.
```

The rendered sentence must remain identical to the current biography contract.

- [ ] **Step 8: Add institution logo assets**

Use the NTU asset from the approved reference commit and the NEAU crest from NEAU's official site asset:

```bash
mkdir -p public/logos
curl -L https://raw.githubusercontent.com/gentlefress/gentlefress.github.io/c9c75ef/logos/NTU.svg -o public/logos/ntu.svg
curl -L https://www.neau.edu.cn/images/logo2.png -o /tmp/neau-logo-source.png
python3 -c "from PIL import Image; im=Image.open('/tmp/neau-logo-source.png').convert('RGBA'); im.crop((0, 0, 79, 79)).save('public/logos/neau.png')"
```

Verify:

```bash
file public/logos/ntu.svg public/logos/neau.png
```

Expected: one SVG and one 79x79 PNG. Do not generate or redraw either institution logo.

- [ ] **Step 9: Run the focused test and commit**

Run:

```bash
npm test
npm run lint
```

Expected: all Task 1 tests PASS and lint exits 0.

Commit:

```bash
git add package.json content/config.toml content/about.toml content/bio.md content/education.toml content/experience.toml public/logos/ntu.svg public/logos/neau.png src/app/page.tsx src/components/home/HomePageClient.tsx tests/homepage-refresh.test.mjs
git commit -m "feat: add GentleFress homepage sections"
```

---

### Task 2: GentleFress Card, News, And Theme Styling

**Files:**
- Modify: `tests/homepage-refresh.test.mjs`
- Modify: `src/components/pages/CardPage.tsx`
- Modify: `src/components/home/News.tsx`
- Modify: `src/lib/stores/themeStore.ts`
- Modify: `src/app/layout.tsx`
- Modify: `src/app/globals.css`

**Interfaces:**
- Consumes: the Task 1 `CardPageConfig` card sections and existing `Theme` store API.
- Produces: conditional image cards, GentleFress-style scrollable News, first-visit light default, and global Morandi-purple accent tokens.

- [ ] **Step 1: Add failing visual and theme source contracts**

Append to `tests/homepage-refresh.test.mjs`:

```js
test('GentleFress visual structure and light-default theme are configured', () => {
  const card = read('src/components/pages/CardPage.tsx');
  const news = read('src/components/home/News.tsx');
  const store = read('src/lib/stores/themeStore.ts');
  const layout = read('src/app/layout.tsx');
  const css = read('src/app/globals.css');

  assert.match(card, /item\.image/);
  assert.match(card, /h-12 w-12/);
  assert.match(news, /max-h-80/);
  assert.match(news, /sm:max-h-96/);
  assert.match(news, /overflow-y-auto/);
  assert.match(news, /ReactMarkdown/);
  assert.match(store, /theme: 'light'/);
  assert.match(layout, /parsed\?\.state\?\.theme \|\| 'light'/);
  assert.match(css, /--accent: #7D6B8C;/);
  assert.match(css, /--accent-dark: #675873;/);
});
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
npm test
```

Expected: FAIL on the missing logo layout, scrollable News, light store default, light bootstrap fallback, and Morandi-purple tokens.

- [ ] **Step 3: Add the conditional GentleFress logo-card layout**

In `src/components/pages/CardPage.tsx`, import `Image`:

```ts
import Image from 'next/image';
```

For each item, keep the existing non-embedded card classes, but use this embedded structure:

```tsx
<motion.div
  key={index}
  initial={false}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.4, delay: 0.1 * index }}
  className={`rounded-lg border border-neutral-200 bg-white shadow-sm transition-shadow duration-200 hover:shadow-md dark:border-neutral-800 dark:bg-neutral-900 ${embedded ? 'flex gap-4 px-4 py-3' : 'p-6'}`}
>
  {item.image && (
    <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center overflow-hidden rounded-lg bg-neutral-50 dark:bg-neutral-800/50">
      <Image
        src={item.image}
        alt=""
        width={32}
        height={32}
        className="h-8 w-8 object-contain"
        aria-hidden="true"
      />
    </div>
  )}
  <div className="min-w-0 flex-1">
    <div className="mb-2 flex min-w-0 flex-col items-start gap-2 sm:flex-row sm:justify-between">
      <h3 className={`${embedded ? 'text-base' : 'text-xl'} min-w-0 break-words font-semibold leading-snug text-primary`}>
        {item.title}
      </h3>
      {item.date && (
        <span className={`${embedded ? 'text-xs' : 'text-sm'} w-fit flex-shrink-0 whitespace-nowrap rounded bg-neutral-100 px-2 py-1 font-medium text-neutral-500 dark:bg-neutral-800`}>
          {item.date}
        </span>
      )}
    </div>
    {item.subtitle && (
      <p className={`${embedded ? 'text-sm' : 'text-base'} mb-2 font-medium text-accent`}>
        {item.subtitle}
      </p>
    )}
    {item.content && (
      <div className={`${embedded ? 'text-sm' : 'text-base'} break-words leading-relaxed text-neutral-600 dark:text-neutral-400`}>
        <ReactMarkdown components={markdownComponents}>{item.content}</ReactMarkdown>
      </div>
    )}
    {item.tags && (
      <div className="mt-4 flex flex-wrap gap-2">
        {item.tags.map((tag) => (
          <span key={tag} className="rounded border border-neutral-100 bg-neutral-50 px-2 py-1 text-xs text-neutral-500 dark:border-neutral-800 dark:bg-neutral-800/50">
            {tag}
          </span>
        ))}
      </div>
    )}
  </div>
</motion.div>
```

Cards without `image`, `date`, `subtitle`, `content`, or `tags` render only their title. Keep `initial={false}`.

- [ ] **Step 4: Replace News markup with the reference list behavior**

Replace `src/components/home/News.tsx` with:

```tsx
'use client';

import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { useMessages } from '@/lib/i18n/useMessages';

export interface NewsItem {
  date: string;
  content: string;
}

interface NewsProps {
  items: NewsItem[];
  title?: string;
}

export default function News({ items, title }: NewsProps) {
  const messages = useMessages();
  const resolvedTitle = title || messages.home.news;

  return (
    <motion.section
      initial={false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.5 }}
    >
      <h2 className="mb-4 font-serif text-3xl font-bold text-primary sm:text-4xl">
        {resolvedTitle}
      </h2>
      <div className="max-h-80 overflow-y-auto overscroll-y-contain scroll-smooth rounded-lg border border-neutral-200/90 bg-neutral-50/60 px-4 py-3 pr-2 sm:max-h-96 dark:border-neutral-700/80 dark:bg-neutral-900/30">
        <ul className="space-y-4 pr-1">
          {items.map((item, index) => (
            <li
              key={`${item.date}-${index}`}
              className="flex flex-col gap-1 border-b border-neutral-200/60 pb-4 last:border-b-0 last:pb-0 sm:flex-row sm:items-start sm:gap-4 dark:border-neutral-700/50"
            >
              <span className="text-sm font-medium tabular-nums text-neutral-500 sm:w-24 sm:flex-shrink-0 sm:text-base">
                {item.date}
              </span>
              <div className="min-w-0 text-base leading-relaxed text-neutral-800 sm:text-lg dark:text-neutral-200">
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <span className="inline">{children}</span>,
                    a: ({ ...props }) => (
                      <a
                        {...props}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-medium text-accent underline-offset-2 hover:underline"
                      />
                    ),
                  }}
                >
                  {item.content}
                </ReactMarkdown>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </motion.section>
  );
}
```

- [ ] **Step 5: Make light the first-visit default**

In `src/lib/stores/themeStore.ts`, change only the initial value and comment:

```ts
      // Default new visitors to light mode.
      theme: 'light',
```

In `src/app/layout.tsx`, change the bootstrap fallback only:

```js
const setting = parsed?.state?.theme || 'light';
```

Keep persisted `light`, `dark`, and `system` handling and the theme toggle unchanged.

- [ ] **Step 6: Replace gold tokens with Morandi purple**

In the light block of `src/app/globals.css`, set:

```css
  /* Brand colors - Deep Navy & Morandi Purple */
  --primary: #1e293b;
  --primary-light: #334155;
  --primary-dark: #0f172a;
  --accent: #7D6B8C;
  --accent-light: #9B8BA8;
  --accent-dark: #675873;
```

In `.dark`, set:

```css
  --accent: #B9A8C7;
  --accent-light: #C9BAD4;
  --accent-dark: #A08EAD;
```

- [ ] **Step 7: Run focused and production checks, then commit**

Run:

```bash
npm test
npm run lint
npm run build
```

Expected: tests, lint, and build all exit 0. The rendered baseline fixture is deliberately updated in Task 3 before running export verification.

Commit:

```bash
git add tests/homepage-refresh.test.mjs src/components/pages/CardPage.tsx src/components/home/News.tsx src/lib/stores/themeStore.ts src/app/layout.tsx src/app/globals.css
git commit -m "feat: match GentleFress homepage styling"
```

---

### Task 3: Confirmed CV Corrections And Export Contract

**Files:**
- Modify: `tests/homepage-refresh.test.mjs`
- Modify: `tests/baseline-content.json`
- Modify: `scripts/verify-export.mjs`
- Modify: `content/cv-json.md`

**Interfaces:**
- Consumes: Task 1 homepage content and Task 2 rendered components.
- Produces: verified rendered-route presence/absence contracts and the three approved CV corrections.

- [ ] **Step 1: Add the failing CV contract**

Append to `tests/homepage-refresh.test.mjs`:

```js
test('CV changes only confirmed stale facts', () => {
  const cv = read('content/cv-json.md');
  assert.match(cv, /\*\*M\.Sc\. Student in Robotics and Intelligent Systems\*\*/);
  assert.doesNotMatch(cv, /Incoming M\.Sc\. Student/);
  assert.match(cv, /### M\.S\. in Robotics and Intelligent Systems/);
  assert.match(cv, /### B\.E\. in Internet of Things/);
  assert.match(cv, /SO-101 Real-World Robotic Learning with LeRobot[\s\S]*?\*\*Robotics Project\*\* · 2026-07-01/);
  assert.match(cv, /Built an imitation-learning pipeline for simulated Franka Panda parcel sorting in ManiSkill3/);
  assert.doesNotMatch(cv, /Developing an imitation-learning solution/);
  assert.match(cv, /under review at IEEE TMM after major revision/);
});
```

- [ ] **Step 2: Run the test and verify it fails only on stale CV facts**

Run:

```bash
npm test
```

Expected: FAIL for `Incoming`, SO-101 `2026-05-01`, and the Marso present-progress sentence. Assertions preserving `M.S.`, `B.E.`, and MSM-Seg status pass.

- [ ] **Step 3: Apply only the approved CV text replacements**

In `content/cv-json.md`:

```markdown
**M.Sc. Student in Robotics and Intelligent Systems**
```

Change the SO-101 date to:

```markdown
**Robotics Project** · 2026-07-01
```

Replace only the Marso summary paragraph with:

```markdown
Built an imitation-learning pipeline for simulated Franka Panda parcel sorting in ManiSkill3, covering color-labeled parcel placement across three difficulty settings, State Diffusion Policy baselines, Easy RGB ACT offline training, and fixed 50-episode evaluation.
```

- [ ] **Step 4: Add route-specific forbidden-text support**

In `scripts/verify-export.mjs`, after the `requiredHtml` loop add:

```js
for (const [route, forbidden] of Object.entries(fixture.forbiddenByRoute || {})) {
  const page = renderedText.get(route) || '';
  for (const text of forbidden) {
    if (page.includes(text)) {
      failures.push(`Forbidden on ${route}: ${text}`);
    }
  }
}
```

- [ ] **Step 5: Update the static-export fixture**

In `tests/baseline-content.json`:

- replace homepage `Biography` with `About`;
- remove all homepage selected-publication-only expectations;
- add the profile title, both Education entries, both exact Experience titles, and their approved supporting Education text;
- replace CV `Incoming M.Sc. Student...` with `M.Sc. Student...`;
- replace CV `2026-05-01` with `2026-07-01`;
- replace the CV Marso present-progress summary with the exact past-tense paragraph from Step 3;
- keep Publications and all MSM-Seg status expectations unchanged;
- add the two logo sources to `requiredHtml.index.html`.

Add this top-level object:

```json
"forbiddenByRoute": {
  "index.html": [
    "Selected Publications"
  ],
  "cv-json/index.html": [
    "Incoming M.Sc. Student in Robotics and Intelligent Systems",
    "Developing an imitation-learning solution with ManiSkill3, SAPIEN, and a Franka Panda manipulator.",
    "2026-05-01"
  ]
}
```

Keep `portfolio/index.html` expectations unchanged because Research is outside this update.

- [ ] **Step 6: Run all automated checks and commit**

Run:

```bash
npm test
npm run lint
npm run check
git diff --check
```

Expected: all commands exit 0; `verify-export` reports five verified routes; no SSR `opacity:0` failure appears.

Commit:

```bash
git add tests/homepage-refresh.test.mjs tests/baseline-content.json scripts/verify-export.mjs content/cv-json.md
git commit -m "fix: refresh confirmed CV facts"
```

---

### Task 4: Browser QA, Documentation, And GitHub Pages Deployment

**Files:**
- Create: `/home/quixoteh/documents/学习相关/obsidian/HH/课业/科研/个人主页/GentleFress主页刷新总结-2026-08-29.md`
- Verify: all files changed in Tasks 1-3

**Interfaces:**
- Consumes: production-ready static export and the existing `.github/workflows/deploy.yml` workflow.
- Produces: desktop/mobile visual evidence, a successful Pages workflow, live-site verification, and a local Obsidian record.

- [ ] **Step 1: Run the clean local verification gate**

Run:

```bash
git status --short --branch
npm test
npm run lint
npm run check
git diff --check
```

Expected: branch is `master`, tests/lint/build/export all exit 0, and the worktree contains only the intended uncommitted documentation if any.

- [ ] **Step 2: Start the local development server**

Run in a persistent terminal session:

```bash
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Expected: Next.js reports `http://127.0.0.1:3000` ready. If port 3000 is occupied, use the next available port and record it.

- [ ] **Step 3: Verify desktop behavior with Playwright**

Using the Playwright skill, open the local URL at `1280x720`, clear `theme-storage`, and reload. Evaluate:

```js
document.documentElement.getAttribute('data-theme') === 'light'
```

Expected: `true`.

Verify visible text and order:

```text
About
Education
Experience
News
```

Verify `Selected Publications` is absent, both institution logos load with non-zero natural dimensions, both Experience cards show only their exact title, and clicking the theme control changes `data-theme` to `dark`. Save `/tmp/quixoteh-home-desktop.png`.

- [ ] **Step 4: Verify mobile layout with Playwright**

Set viewport to `390x844`, return the theme to light, and reload. Evaluate:

```js
document.documentElement.scrollWidth <= document.documentElement.clientWidth
```

Expected: `true`.

Open the mobile navigation, confirm all five links fit and work, inspect Education date wrapping, verify Experience remains title-only, scroll News, and save `/tmp/quixoteh-home-mobile.png`.

- [ ] **Step 5: Inspect screenshots and console**

Open both PNGs with the local image viewer. Confirm:

- no text overlap or clipped date badges;
- no broken or invisible logo;
- Morandi purple is used for links/profile title/institution subtitles;
- page is not dominated by purple;
- News scroll container is readable;
- browser console has no site-caused error or hydration warning.

If any check fails, fix only the responsible file, rerun Tasks 2-3 checks, and create a focused commit before continuing.

- [ ] **Step 6: Push and watch the deployment**

Run:

```bash
git status --short --branch
git push origin master
gh run list --workflow deploy.yml --limit 1 --json databaseId,status,conclusion,headSha,url
run_id="$(gh run list --workflow deploy.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
gh run watch "$run_id" --exit-status
```

Expected: the latest run targets the pushed HEAD and finishes with `conclusion: success`.

- [ ] **Step 7: Verify the uncached live site**

Run:

```bash
curl -fsSL https://quixoteh.github.io/ | rg 'M.Sc. Student in Robotics and Intelligent Systems|Education|A\*STAR Research Attachment|News'
```

Expected: all four strings appear in the live HTML.

Use Playwright on `https://quixoteh.github.io/` with fresh storage at desktop and mobile widths. Confirm the page is light, nonblank, contains no `Selected Publications`, and matches the locally approved screenshots.

- [ ] **Step 8: Write the Obsidian implementation record**

Collect the literal evidence first:

```bash
git log --format='%h %s' 49d8f3a..HEAD
gh run view "$run_id" --json url,headSha,conclusion --jq '{url,headSha,conclusion}'
```

Then use `apply_patch` to create the summary with the returned commit hashes and workflow URL written literally. Use this fixed structure:

```markdown
# GentleFress 主页刷新总结（2026-08-29）

## 基准

- 设计提交：`49d8f3a`
- 实施提交：Tasks 1-3 的实际提交哈希与提交主题
- 线上地址：https://quixoteh.github.io/
- GitHub Actions：成功 workflow 的实际 URL

## 已实施

- 默认亮色与莫兰迪紫强调色
- `About / Research / Publications / Awards / CV` 导航
- `About / Education / Experience / News` 首页结构
- Experience 两个严格单行条目
- 已确认的 CV 时态与日期修正；`M.S.`、`B.E.` 和 MSM-Seg 状态未改

## 验证

- `npm test`
- `npm run lint`
- `npm run check`
- `git diff --check`
- Playwright：1280x720 与 390x844
- GitHub Pages workflow 成功
- 正式网址无缓存参数验证通过
```

The finished record must contain the actual hashes and workflow URL. Verify it with `rg -n 'TBD|TODO|填写|placeholder'` and require no matches.

---

## Plan Self-Review

- Spec coverage: navigation, homepage order, strict single-line Experience, Education facts/logos, News layout, light default, Morandi purple, CV boundaries, SSR visibility, browser QA, deployment, and Obsidian backup all have explicit tasks.
- Placeholder scan: deployment identifiers are captured into shell variables at runtime; the final Obsidian record explicitly rejects unresolved marker text.
- Type consistency: both homepage `SectionConfig` definitions use the same `config?: CardPageConfig`; TOML card files match the existing `CardPageConfig` and `CardItem` fields.
