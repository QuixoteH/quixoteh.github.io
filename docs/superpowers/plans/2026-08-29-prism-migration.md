# PRISM Website Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Jekyll/AcademicPages site with a statically exported PRISM Next.js site while preserving every public text value from GitHub commit `60aaebf`.

**Architecture:** Adopt PRISM commit `f2748db` as the application foundation, keep public content in TOML/Markdown/BibTeX files, and deploy the static `out/` export through GitHub Actions. A content contract tied to the `60aaebf` snapshot protects wording while the visual components and build system change.

**Tech Stack:** Next.js 15.3.3, React 19, TypeScript, Tailwind CSS 4, TOML, Markdown, BibTeX, Node.js 22, GitHub Actions, GitHub Pages.

## Global Constraints

- Content baseline is exactly `QuixoteH/quixoteh.github.io` commit `60aaebf`.
- Template baseline is exactly `xyjoey/PRISM` commit `f2748db`.
- Preserve the public labels and paths `/`, `/portfolio/`, `/publications/`, `/teaching/`, and `/cv-json/`.
- Preserve Biography, Interests, News, Research, Publications, Awards, CV, profile identity, and public social values without copy editing.
- Allow only format escaping required by TOML, Markdown, BibTeX, or generated HTML.
- Reuse the existing `images/profile.png`; do not create or alter a portrait.
- Keep light/dark theme switching; disable i18n, language switching, likes, analytics, forms, blog, Teaching, and Services.
- Store maintainable source in `master`; do not commit `out/` or `node_modules/`.
- Back up the design, plan, and final migration record under `/home/quixoteh/documents/学习相关/obsidian/HH/课业/科研/个人主页/`.

---

## File Structure

### Retained PRISM application files

- `src/app/`: static routes, metadata, layout, and global styles.
- `src/components/`: PRISM navigation, profile, publication, card, text, theme, and footer components.
- `src/lib/`: TOML, Markdown, BibTeX, configuration, and i18n-disabled runtime loaders.
- `src/types/`: content and component contracts.
- `content/`: all user-visible migrated copy and page configuration.
- `public/`: profile image, project/publication media, favicon, and compatibility redirects.

### Project-specific files

- `content/config.toml`: identity, visible social links, feature flags, and preserved navigation URLs.
- `content/about.toml`, `content/bio.md`, `content/news.toml`: homepage content.
- `content/portfolio.toml`: five Research cards.
- `content/publications.toml`, `content/publications.bib`: MSM-Seg page and selected homepage entry.
- `content/teaching.toml`: Kaggle award card.
- `content/cv-json.toml`, `content/cv-json.md`: current structured CV rendered as Markdown.
- `tests/baseline-content.json`: content assertions transcribed from commit `60aaebf`.
- `scripts/verify-export.mjs`: verifies exported routes, required copy, banned examples, and local asset references.
- `.github/workflows/deploy.yml`: build, verify, and Pages deployment.

---

### Task 1: Add A Failing Baseline Content Contract

**Files:**
- Create: `tests/baseline-content.json`
- Create: `scripts/verify-export.mjs`
- Modify: `package.json`

**Interfaces:**
- Consumes: generated static files under `out/`.
- Produces: `npm run verify:export`, exiting zero only when all preserved routes and strings exist and PRISM example copy is absent.

- [ ] **Step 1: Create the baseline fixture**

Create `tests/baseline-content.json` with this schema and populate every array from commit `60aaebf` without rewriting:

```json
{
  "routes": ["index.html", "portfolio/index.html", "publications/index.html", "teaching/index.html", "cv-json/index.html"],
  "required": {
    "index.html": [
      "Hai HUANG",
      "Nanyang Technological University",
      "Singapore",
      "Hai Huang (Student Member, IEEE) received the B.Eng. degree in Internet of Things Engineering from Northeast Agricultural University, Harbin, China, in 2026.",
      "Force-aware Robot Learning",
      "MSM-Seg was accepted for publication as a regular paper in IEEE Transactions on Multimedia."
    ],
    "portfolio/index.html": [
      "MSM-Seg: Multi-Modal Brain Tumor Segmentation",
      "Marso Hack Berlin 2026 -- Robot Parcel Sorting Challenge",
      "MuJoCo Playground -- Vision-Based RL for Panda Arm Grasping",
      "YOLOv8 and IoT Agricultural Pest Detection System",
      "SO-101 Real-World Robotic Learning with LeRobot"
    ],
    "publications/index.html": [
      "MSM-Seg: A Modality-and-Slice Memory Framework with Category-Agnostic Prompting for Multi-Modal Brain Tumor Segmentation",
      "Yuxiang Luo, Qing Xu, Hai Huang, et al.",
      "arXiv:2510.10679"
    ],
    "teaching/index.html": [
      "Silver Medal -- Home Credit: Credit Risk Model Stability",
      "ranking in the top 0.5%"
    ],
    "cv-json/index.html": [
      "AI Solutions Engineer Intern",
      "China Unicom Chengdu Branch, Digital Technology Center",
      "M.S. in Robotics and Intelligent Systems",
      "IELTS 7.0"
    ]
  },
  "banned": [
    "Jiale Liu",
    "University of Example",
    "Computational Physics",
    "Example Award",
    "john.doe@example.com"
  ]
}
```

The checked-in fixture must extend the representative values shown above with every visible string from these exact baseline sources: `_pages/about.md`, all five `_portfolio/*.md` files, `_publications/2025-msm-seg.md`, `_teaching/2024-kaggle-home-credit.md`, and every non-empty public value in `_data/cv.json`. Keep complete sentences as single array elements so the verifier rejects wording changes rather than only checking isolated keywords.

- [ ] **Step 2: Add the export verifier**

Create `scripts/verify-export.mjs`:

```javascript
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
  .replace(/&lt;/g, '<')
  .replace(/&gt;/g, '>')
  .replace(/\s+/g, ' ')
  .trim();

const failures = [];
const rendered = new Map();

for (const route of fixture.routes) {
  const file = path.join(root, 'out', route);
  if (!fs.existsSync(file)) {
    failures.push(`Missing route: ${route}`);
    continue;
  }
  rendered.set(route, decodeHtml(fs.readFileSync(file, 'utf8')));
}

for (const [route, required] of Object.entries(fixture.required)) {
  const page = rendered.get(route) || '';
  for (const text of required) {
    if (!page.includes(text)) failures.push(`Missing from ${route}: ${text}`);
  }
}

const corpus = [...rendered.values()].join(' ');
for (const text of fixture.banned) {
  if (corpus.includes(text)) failures.push(`Example content remains: ${text}`);
}

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(`Verified ${fixture.routes.length} routes and preserved content.`);
```

- [ ] **Step 3: Add the script entry**

After PRISM's `package.json` is present, add:

```json
"verify:export": "node scripts/verify-export.mjs",
"check": "npm run build && npm run verify:export"
```

- [ ] **Step 4: Run the verifier before migration**

Run: `node scripts/verify-export.mjs`

Expected: non-zero exit with all five `Missing route` messages because `out/` does not yet contain the PRISM export.

- [ ] **Step 5: Commit the contract**

```bash
git add tests/baseline-content.json scripts/verify-export.mjs
git commit -m "test: capture public site content contract"
```

---

### Task 2: Replace Jekyll With The PRISM Application Foundation

**Files:**
- Remove: tracked Jekyll runtime files and directories made obsolete by the migration.
- Create from PRISM `f2748db`: `.nvmrc`, `.gitignore`, `eslint.config.mjs`, `next.config.ts`, `package.json`, `package-lock.json`, `postcss.config.mjs`, `tailwind.config.mjs`, `tsconfig.json`, `src/**`.
- Create: `LICENSE-PRISM`
- Preserve: `docs/superpowers/**`, `.git/`, and content/media sources until Task 3 finishes.

**Interfaces:**
- Consumes: PRISM source tree at `/tmp/prism-template-20260829`.
- Produces: installable Next.js source with no migrated content yet.

- [ ] **Step 1: Copy the PRISM foundation mechanically**

Use exact source files from commit `f2748db`:

```bash
cp -a /tmp/prism-template-20260829/.nvmrc .
cp -a /tmp/prism-template-20260829/.gitignore .
cp -a /tmp/prism-template-20260829/eslint.config.mjs .
cp -a /tmp/prism-template-20260829/next.config.ts .
cp -a /tmp/prism-template-20260829/package.json .
cp -a /tmp/prism-template-20260829/package-lock.json .
cp -a /tmp/prism-template-20260829/postcss.config.mjs .
cp -a /tmp/prism-template-20260829/tailwind.config.mjs .
cp -a /tmp/prism-template-20260829/tsconfig.json .
cp -a /tmp/prism-template-20260829/src .
cp -a /tmp/prism-template-20260829/LICENSE LICENSE-PRISM
```

- [ ] **Step 2: Remove template coupling**

In `src/app/globals.css`, remove the remote `jialeliu.com` font face and use:

```css
--font-serif: Georgia, 'Times New Roman', serif;
```

In `src/app/layout.tsx`, remove the DNS prefetch, preconnect, and font preload tags for `jialeliu.com`. Keep the local system font stack and existing theme bootstrap.

- [ ] **Step 3: Install dependencies**

Run: `npm ci`

Expected: exit zero using Node.js 22 and the committed lockfile.

- [ ] **Step 4: Confirm the unconfigured template builds**

Copy PRISM's temporary example `content/` and `public/favicon.svg` only for this check, then run `npm run build`.

Expected: Next.js exits zero and creates `out/`.

Remove the temporary example `content/` before Task 3 so no example copy can survive.

- [ ] **Step 5: Commit the application foundation**

```bash
git add .nvmrc .gitignore eslint.config.mjs next.config.ts package.json package-lock.json postcss.config.mjs tailwind.config.mjs tsconfig.json src LICENSE-PRISM
git commit -m "build: replace Jekyll runtime with PRISM"
```

---

### Task 3: Migrate Homepage And Profile Content

**Files:**
- Create: `content/config.toml`
- Create: `content/about.toml`
- Create: `content/bio.md`
- Create: `content/news.toml`
- Copy: `images/profile.png` to `public/profile.png`
- Copy: `images/favicon.svg` to `public/favicon.svg`
- Modify: `src/components/home/Profile.tsx`
- Modify: `src/components/layout/Footer.tsx`

**Interfaces:**
- Consumes: baseline `_config.yml`, `_pages/about.md`, `images/profile.png`, and PRISM homepage component contracts.
- Produces: exact online homepage content in the PRISM layout with theme switching only.

- [ ] **Step 1: Configure identity, features, and preserved navigation**

Create `content/config.toml` with these structural values and exact baseline identity/social values:

```toml
[site]
title = "Hai HUANG"
description = "Robotics and intelligent systems researcher"
favicon = "/favicon.svg"
last_updated = "August 29, 2026"

[author]
name = "Hai HUANG"
title = ""
institution = "Nanyang Technological University"
avatar = "/profile.png"

[social]
location = "Singapore"
github = "https://github.com/QuixoteH"
linkedin = "https://www.linkedin.com/in/quixoteh"

[features]
enable_likes = false
enable_one_page_mode = false

[i18n]
enabled = false
locales = ["en"]
default_locale = "en"
mode = "fixed"
fixed_locale = "en"
persist = false
switcher = false

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

- [ ] **Step 2: Create the homepage section configuration**

Create `content/about.toml`:

```toml
type = "about"
title = "Biography"

[profile]
research_interests = [
  "Force-aware Robot Learning",
  "Robot Manipulation",
  "Vision-Language-Action Models"
]

[[sections]]
id = "biography"
type = "markdown"
source = "bio.md"
title = "Biography"

[[sections]]
id = "featured_publications"
type = "publications"
title = "Selected Publications"
filter = "selected"
limit = 1

[[sections]]
id = "news"
type = "list"
title = "News"
source = "news.toml"
```

- [ ] **Step 3: Copy biography and news exactly**

Put the single Biography paragraph from `_pages/about.md` into `content/bio.md` without edits. Convert each News bullet to one `[[news]]` object while preserving its displayed date and complete sentence:

```toml
[[news]]
date = "Aug 2026"
content = "MSM-Seg was accepted for publication as a regular paper in IEEE Transactions on Multimedia."
```

Repeat in source order for all seven entries.

- [ ] **Step 4: Keep empty profile title truly empty**

Adjust `Profile.tsx` so `author.title` is rendered only when non-empty. Keep institution, location, GitHub, LinkedIn, and research interests. Remove the like-state code path from visible output because `enable_likes` is false.

- [ ] **Step 5: Copy only requested assets**

```bash
mkdir -p public
cp images/profile.png public/profile.png
cp images/favicon.svg public/favicon.svg
```

Verify `sha256sum images/profile.png public/profile.png` prints the same digest twice.

- [ ] **Step 6: Build and check homepage content**

Run: `npm run build && node scripts/verify-export.mjs`

Expected at this phase: homepage assertions pass; missing Research, Publications, Awards, and CV assertions remain.

- [ ] **Step 7: Commit homepage migration**

```bash
git add content public/profile.png public/favicon.svg src/components/home/Profile.tsx src/components/layout/Footer.tsx package.json
git commit -m "feat: migrate profile and homepage to PRISM"
```

---

### Task 4: Migrate Research, Publications, Awards, And CV

**Files:**
- Create: `content/portfolio.toml`
- Create: `content/publications.toml`
- Create: `content/publications.bib`
- Create: `content/teaching.toml`
- Create: `content/cv-json.toml`
- Create: `content/cv-json.md`
- Modify: `src/components/pages/CardPage.tsx`
- Modify: `src/components/pages/TextPage.tsx`
- Modify: `src/components/publications/PublicationsList.tsx` only if exact baseline fields cannot be displayed by the template.

**Interfaces:**
- Consumes: five `_portfolio/*.md` files, `_publications/2025-msm-seg.md`, `_teaching/2024-kaggle-home-credit.md`, and `_data/cv.json` from commit `60aaebf`.
- Produces: four PRISM routes containing every baseline field and no invented copy.

- [ ] **Step 1: Convert Research entries to cards**

Create `content/portfolio.toml` with `type = "card"`, `title = "Research"`, and five `[[items]]` entries in reverse chronological order. Map fields mechanically:

```toml
[[items]]
title = "Marso Hack Berlin 2026 -- Robot Parcel Sorting Challenge"
subtitle = "Imitation learning for simulated parcel sorting with ManiSkill3, SAPIEN, and a Franka Panda manipulator."
date = "Jul 2026"
content = """
- Developing an imitation-learning solution for a simulated robot parcel-sorting challenge using ManiSkill3, SAPIEN, and a Franka Panda manipulator.
- Parsing and replaying expert demonstration trajectories stored in HDF5/JSON formats across easy, medium, and hard task settings; extracting observations, actions, environment states, and success labels for training and evaluation.
- Establishing behavior-cloning baselines from state and RGB observations, and evaluating advanced visuomotor policies such as Diffusion Policy or ACT to improve generalization across held-out test episodes.
- Building a reproducible training, rollout, and submission pipeline; measuring policy performance by task success rate and preparing competition submissions on Kaggle.
"""
```

Repeat the same front-matter/body mapping for SO-101, MuJoCo Playground, pest detection, and MSM-Seg without changing any sentence.

- [ ] **Step 2: Convert MSM-Seg to BibTeX without changing claims**

Create `content/publications.toml`:

```toml
type = "publication"
title = "Publications"
source = "publications.bib"
```

Create one BibTeX entry whose title, authors, year, venue/status, description, and URL reproduce `_publications/2025-msm-seg.md`. Set `selected = {true}` so the same entry appears on the homepage. Do not replace the publication page's baseline `under review` statement with the newer homepage acceptance announcement because the baseline contains both claims on different pages.

- [ ] **Step 3: Convert the award to a card**

Create `content/teaching.toml`:

```toml
type = "card"
title = "Awards"

[[items]]
title = "Silver Medal -- Home Credit: Credit Risk Model Stability"
subtitle = "Kaggle Competition Award | Kaggle"
date = "Apr 2024"
content = "Awarded a silver medal in the Kaggle competition **Home Credit -- Credit Risk Model Stability**, ranking in the top 0.5%."
```

- [ ] **Step 4: Convert the JSON CV to Markdown mechanically**

Create `content/cv-json.toml`:

```toml
type = "text"
title = "CV"
source = "cv-json.md"
```

Create `content/cv-json.md` with the baseline section order and every non-empty `_data/cv.json` value: Basics, Work Experience, Education, Skills, Languages, Interests, Publications, and Research Projects. Do not render the empty phone, references, presentations, or teaching arrays. Preserve all dates, summaries, highlights, course names, URLs, and project descriptions exactly.

- [ ] **Step 5: Stabilize card and CV layouts**

Keep card border radius at `rounded-lg`, use a fixed date column on desktop and wrapped date line on mobile, and ensure long project titles wrap instead of colliding with dates. In `TextPage.tsx`, use outside list markers (`list-disc`) with `pl-5`, and ensure long URLs wrap with `break-words`.

- [ ] **Step 6: Run the complete content contract**

Run: `npm run check`

Expected: Next.js build exits zero and `Verified 5 routes and preserved content.` appears.

- [ ] **Step 7: Commit migrated content pages**

```bash
git add content src/components/pages src/components/publications tests scripts package.json
git commit -m "feat: migrate research publications awards and CV"
```

---

### Task 5: Add Compatibility Routes And GitHub Pages Deployment

**Files:**
- Create: `scripts/create-redirects.mjs`
- Modify: `package.json`
- Create: `.github/workflows/deploy.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: preserved legacy route map and successful static export.
- Produces: redirect HTML in `out/` and an automated Pages deployment from `master`.

- [ ] **Step 1: Generate legacy redirects after the Next build**

Create `scripts/create-redirects.mjs` with a route map for `/about/`, `/about.html`, five old project detail URLs, `/publication/2025-msm-seg/`, and `/award/2024-kaggle-home-credit/`. Each generated HTML document must contain a canonical link, zero-second meta refresh, and visible fallback link to the retained destination.

Use these destinations:

```javascript
const redirects = {
  'about/index.html': '/',
  'about.html': '/',
  'portfolio/2025-msm-seg/index.html': '/portfolio/',
  'portfolio/2026-marso/index.html': '/portfolio/',
  'portfolio/2026-mujoco-playground/index.html': '/portfolio/',
  'portfolio/2026-pest-detection/index.html': '/portfolio/',
  'portfolio/2026-so101/index.html': '/portfolio/',
  'publication/2025-msm-seg/index.html': '/publications/',
  'award/2024-kaggle-home-credit/index.html': '/teaching/'
};
```

- [ ] **Step 2: Run redirects as part of the build**

Change package scripts to:

```json
"build": "next build && node scripts/create-redirects.mjs",
"verify:export": "node scripts/verify-export.mjs",
"check": "npm run build && npm run verify:export"
```

- [ ] **Step 3: Add the deployment workflow**

Create `.github/workflows/deploy.yml` that triggers on pushes to `master` and manual dispatch, grants only `contents: read`, `pages: write`, and `id-token: write`, runs `npm ci`, `npm run check`, uploads `out/` with `actions/upload-pages-artifact@v3`, and deploys with `actions/deploy-pages@v4` in a separate `github-pages` environment job.

- [ ] **Step 4: Replace template documentation**

Create a concise `README.md` containing the site URL, local commands (`npm ci`, `npm run dev`, `npm run check`), content file map, and deployment behavior. Credit PRISM and link its MIT license without retaining example-user instructions.

- [ ] **Step 5: Verify redirects and workflow syntax**

Run:

```bash
npm run check
test -f out/about/index.html
test -f out/portfolio/2026-so101/index.html
test -f out/publication/2025-msm-seg/index.html
test -f out/award/2024-kaggle-home-credit/index.html
```

Expected: every command exits zero.

- [ ] **Step 6: Commit deployment support**

```bash
git add .github README.md package.json scripts
git commit -m "ci: deploy PRISM site to GitHub Pages"
```

---

### Task 6: Remove Obsolete Jekyll Files And Perform Local Acceptance

**Files:**
- Remove: Jekyll-only tracked files and directories after verifying their content/assets have migrated.
- Preserve: `docs/superpowers/**`, `.github/`, PRISM source, migrated content, required public assets, tests, and scripts.
- Create: `/home/quixoteh/documents/学习相关/obsidian/HH/课业/科研/个人主页/PRISM主页迁移总结-2026-08-29.md`

**Interfaces:**
- Consumes: complete PRISM build and migrated assets.
- Produces: minimal maintainable source tree and local browser acceptance evidence.

- [ ] **Step 1: Audit retained asset references**

Search all migrated content and code for `/profile.png`, project media, paper links, and favicon references. Copy any referenced local assets into `public/` before deleting their old Jekyll directories.

- [ ] **Step 2: Remove obsolete Jekyll sources**

Use `git rm` only for tracked files made obsolete by the migration, including `_config.yml`, `_layouts/`, `_includes/`, `_sass/`, `_pages/`, `_portfolio/`, `_publications/`, `_teaching/`, unused sample posts/talks/data, Jekyll assets, `Gemfile`, Docker/Jekyll helpers, and old AcademicPages documentation. Do not remove the already committed design and implementation plan.

- [ ] **Step 3: Run source and build hygiene checks**

Run:

```bash
npm ci
npm run check
rg -n "Jiale Liu|University of Example|Example Award|john.doe@example.com" content public src out
git diff --check
git status --short
```

Expected: install and check exit zero; `rg` returns no matches; `git diff --check` returns no output.

- [ ] **Step 4: Start the production-equivalent static server**

Run: `npx serve out -l 4173`

Expected: server remains available at `http://127.0.0.1:4173` for browser acceptance.

- [ ] **Step 5: Check every page visually**

At desktop `1280x720` and mobile `390x844`, inspect:

- `/`
- `/portfolio/`
- `/publications/`
- `/teaching/`
- `/cv-json/`

For each viewport, confirm header/menu operation, profile framing, wrapped text, card/date alignment, no overlap, no horizontal overflow, light/dark rendering, theme persistence, working internal navigation, and successful profile/favicon loading. Check console logs after navigating all pages and require zero site errors and zero site warnings.

- [ ] **Step 6: Record acceptance evidence in Obsidian**

Create the migration summary with baseline/template commits, final branch and commit list, commands and results, checked routes/viewports/themes, deployment workflow name, public URL, and any intentionally preserved content inconsistency such as the baseline publication page's older status text.

- [ ] **Step 7: Commit cleanup and acceptance record references**

```bash
git add -A
git commit -m "chore: remove obsolete Jekyll site"
```

---

### Task 7: Push, Deploy, And Verify The Public Site

**Files:**
- No new source files expected unless deployment verification reveals a defect.

**Interfaces:**
- Consumes: clean, locally accepted `codex/migrate-to-prism` branch.
- Produces: merged `master`, successful GitHub Pages workflow, and verified public PRISM site.

- [ ] **Step 1: Re-run the full pre-push gate**

Run:

```bash
npm ci
npm run check
git diff --check origin/master...HEAD
git status --short --branch
```

Expected: commands exit zero; working tree is clean; branch is ahead of `origin/master` only by migration commits.

- [ ] **Step 2: Push the migration branch**

Run: `git push -u origin codex/migrate-to-prism`

Expected: remote branch is created successfully.

- [ ] **Step 3: Merge to master without rewriting history**

Fast-forward or merge the reviewed migration branch into the latest `master`, then run `npm run check` again before pushing `master`. If remote `master` changed, stop and rebase/merge those user changes before continuing; never force-push.

- [ ] **Step 4: Configure GitHub Pages for Actions**

Use the authenticated GitHub API to set the Pages build source to `workflow` if it is not already configured. Confirm the repository remains public and the Pages URL remains `https://quixoteh.github.io/`.

- [ ] **Step 5: Push master and watch deployment**

Run `git push origin master`, then use `gh run list` and `gh run watch` for the `Deploy PRISM to GitHub Pages` workflow.

Expected: build and deploy jobs both conclude `success`.

- [ ] **Step 6: Verify the live site**

Open the public URL with a cache-busting query and repeat the desktop/mobile checks for homepage, navigation routes, profile image, theme toggle, legacy redirects, and console output. Confirm the served HTML is the PRISM export rather than the former Jekyll page.

- [ ] **Step 7: Update the Obsidian migration summary**

Record the final master commit, workflow run URL/result, deployment timestamp, public verification result, and exact commands used for the final gate.
