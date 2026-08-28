# Profile Email And Experience Logos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the approved homepage name/title changes, email icon, official Experience logos, and homepage-only GPA removal without changing unrelated content.

**Architecture:** Keep `content/config.toml`, `content/education.toml`, and `content/experience.toml` as the data owners. Extend the existing profile social-link array with Lucide `Mail`, and reuse `CardPage` image rendering for both Experience logos. Store local logo assets derived from official organization-hosted files so the static site has no runtime third-party dependency.

**Tech Stack:** Next.js 15, React 19, TypeScript, TOML content, Lucide React, Node test runner, Playwright browser QA, GitHub Pages.

## Global Constraints

- Homepage author heading: `Hai Huang`.
- Homepage profile title: `M.S. Student in Robotics`.
- Homepage biography section heading: `Biography`; navigation label remains `About`.
- Email target: `mailto:quixotehh@gmail.com`; accessible name and hover title: `Email`.
- Experience titles remain exactly `A*STAR Research Attachment` and `China Unicom AI Solutions Engineer Intern`.
- Experience entries contain only `title` and `image`.
- Homepage Education removes `GPA: 85/100`; CV retains it.
- Biography, News, Research, Publications, Awards, CV, navigation, theme, accent colors, and profile institution remain unchanged.
- Logo artwork comes from official A*STAR and China Unicom websites and is not redrawn or recolored.

---

### Task 1: Lock Approved Content And Rendering Contracts

**Files:**
- Modify: `tests/homepage-refresh.test.mjs`
- Modify: `tests/baseline-content.json`

**Interfaces:**
- Consumes: TOML content files, `Profile.tsx`, and exported homepage HTML.
- Produces: Exact automated contracts for the approved profile text, email link, GPA boundary, and logo paths.

- [ ] **Step 1: Write the failing source/content assertions**

Update the homepage configuration assertions to require:

```js
assert.equal(config.author.name, 'Hai Huang');
assert.equal(config.author.title, 'M.S. Student in Robotics');
assert.equal(config.social.email, 'quixotehh@gmail.com');
```

Replace the Education and Experience expectations with:

```js
assert.deepEqual(education.items[1], {
  title: 'B.E. in Internet of Things',
  subtitle: 'Northeast Agricultural University',
  date: 'Sep. 2022 - Jun. 2026',
  content: 'College of Intelligent Science and Engineering',
  image: '/logos/neau.png',
});

assert.deepEqual(experience.items, [
  { title: 'A*STAR Research Attachment', image: '/logos/astar.png' },
  {
    title: 'China Unicom AI Solutions Engineer Intern',
    image: '/logos/china-unicom.png',
  },
]);
for (const item of experience.items) {
  assert.deepEqual(Object.keys(item), ['title', 'image']);
}
```

Add assertions that `content/cv-json.md` still contains `85/100`, `Profile.tsx` imports `Mail`, and both logo files exist with non-zero size.

- [ ] **Step 2: Update the exported HTML baseline**

In `tests/baseline-content.json`, require `Hai Huang` and `M.S. Student in Robotics` on `index.html`, remove homepage `GPA: 85/100`, add it to the homepage forbidden list, retain CV `85/100`, and add:

```json
"href=\"mailto:quixotehh@gmail.com\"",
"src=\"/logos/astar.png\"",
"src=\"/logos/china-unicom.png\""
```

- [ ] **Step 3: Run the focused tests and confirm the red state**

Run:

```bash
npm test
```

Expected: FAIL because the current profile/config values, GPA, Experience image fields, email link, and logo files do not yet satisfy the new contracts.

---

### Task 2: Add Official Assets And Minimal Homepage Changes

**Files:**
- Create: `public/logos/astar.png`
- Create: `public/logos/china-unicom.png`
- Modify: `content/config.toml`
- Modify: `content/about.toml`
- Modify: `content/education.toml`
- Modify: `content/experience.toml`
- Modify: `src/components/home/Profile.tsx`
- Modify: `src/components/pages/CardPage.tsx`

**Interfaces:**
- Consumes: `SiteConfig.social.email`, Lucide `Mail`, and `CardItem.image`.
- Produces: An icon-only `mailto:` control and local Experience logo paths rendered by existing `CardPage` logic.

- [ ] **Step 1: Prepare official local logo assets**

Download the A*STAR favicon from the official A*STAR site and export its largest embedded frame as PNG:

```bash
curl -L --fail --silent --show-error \
  'https://www.a-star.edu.sg/api/media/867ab78f-c433-4489-9de0-51f3937f384c/favicon.ico' \
  -o /tmp/astar-favicon.ico
ffmpeg -y -i /tmp/astar-favicon.ico -frames:v 1 public/logos/astar.png
```

Download the official China Unicom Global wordmark, isolate its existing symbol without recoloring it, and place the symbol on a transparent square canvas:

```bash
curl -L --fail --silent --show-error \
  'https://www.chinaunicomglobal.com/cunicms/uplus/unicom-logo-black.png' \
  -o /tmp/china-unicom-official.png
node scripts/extract-unicom-logo.mjs
```

The temporary `scripts/extract-unicom-logo.mjs` uses the existing `sharp` package to extract `{ left: 0, top: 0, width: 1500, height: 1138 }`, trim transparent padding, and resize with `fit: 'contain'` to a transparent `256 x 256` PNG. Delete it immediately with `apply_patch` because it is a one-use asset preparation step.

- [ ] **Step 2: Apply the approved TOML changes**

Set these exact values in `content/config.toml`:

```toml
[author]
name = "Hai Huang"
title = "M.S. Student in Robotics"

[social]
email = "quixotehh@gmail.com"
```

Change only the homepage markdown section title in `content/about.toml`:

```toml
[[sections]]
id = "about"
type = "markdown"
source = "bio.md"
title = "Biography"
```

Change the NEAU Education content to:

```toml
content = "College of Intelligent Science and Engineering"
```

Set Experience items to:

```toml
[[items]]
title = "A*STAR Research Attachment"
image = "/logos/astar.png"

[[items]]
title = "China Unicom AI Solutions Engineer Intern"
image = "/logos/china-unicom.png"
```

- [ ] **Step 3: Add the email icon to the existing social-link loop**

Import `Mail` from `lucide-react`. Add the email item before GitHub and LinkedIn with `href: \`mailto:${social.email}\`` and `external: false`; mark GitHub and LinkedIn as `external: true`. Render `target` and `rel` only when `external` is true, leaving the existing icon button classes, `aria-label`, title, and screen-reader text unchanged.

For embedded cards that contain only `title` and `image`, use compact mobile spacing, a `40px` logo container, and `text-xs`; prevent wrapping from `360px` upward and retain the existing sizes from the `sm` breakpoint upward. Below `360px`, allow natural wrapping instead of horizontal overflow. This preserves both Experience titles on one line at the verified `390px` viewport without changing Education cards.

- [ ] **Step 4: Run the focused tests and confirm the green state**

Run:

```bash
npm test
```

Expected: all Node tests PASS.

- [ ] **Step 5: Commit the implementation**

```bash
git add content/config.toml content/education.toml content/experience.toml \
  src/components/home/Profile.tsx public/logos/astar.png \
  public/logos/china-unicom.png tests/homepage-refresh.test.mjs \
  tests/baseline-content.json
git commit -m "feat: add profile contact and experience logos"
```

---

### Task 3: Verify, Deploy, And Inspect The Live Page

**Files:**
- Verify only: repository test/build output and exported `out/` files.

**Interfaces:**
- Consumes: completed implementation commit.
- Produces: verified local static export and successful GitHub Pages deployment.

- [ ] **Step 1: Run all repository checks**

Run:

```bash
npm test
npm run lint
npm run build
npm run verify:export
git diff --check
```

Expected: every command exits `0`; exported content includes the mail link and both logo paths, retains CV GPA, and excludes homepage GPA.

- [ ] **Step 2: Perform desktop and mobile browser QA**

Serve the export and inspect `1280x720` and `390x844` viewports with Playwright. Verify the three social icons, both Experience logos, exact profile text, no homepage GPA, no title wrapping regression, no blank or hidden content, and no overflow/overlap. Save screenshots under `/tmp` rather than committing them.

- [ ] **Step 3: Push and monitor deployment**

```bash
git push origin master
run_id=$(gh run list --workflow deploy.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$run_id" --exit-status
```

Expected: the newest deployment workflow completes successfully.

- [ ] **Step 4: Verify the uncached live homepage**

Set `commit=$(git rev-parse HEAD)`, open `https://quixoteh.github.io/?verify=$commit`, and repeat the key desktop/mobile assertions. Confirm that the fetched page references the pushed commit's exact text and assets.
