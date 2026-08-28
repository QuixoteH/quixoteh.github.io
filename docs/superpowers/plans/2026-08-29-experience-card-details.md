# Experience Card Details Implementation Plan

> **For agentic workers:** Execute this plan task-by-task in the current primary-model session.

**Goal:** Populate both Experience cards with Zhe Li-style metadata, replace the blurry A*STAR icon, and replace the heart favicon with a simple book.

**Architecture:** Use the existing `CardItem` fields and standard `CardPage` rendering. Remove the obsolete title-only responsive branch after the Experience data gains subtitle, date, content, and tags.

**Tech Stack:** TOML, React/TypeScript, Next.js 15, Node test runner, Playwright, GitHub Pages.

## Task 1: Lock The New Contract

- [ ] Update `tests/homepage-refresh.test.mjs` to require the exact two Experience objects from the design.
- [ ] Require every Experience item to contain `title`, `subtitle`, `date`, `content`, `image`, and `tags` in that order.
- [ ] Require `public/logos/astar.png` to report PNG dimensions `675 x 675`.
- [ ] Require `content/config.toml` to use `/favicon-book.svg` and verify the SVG contains the Lucide `BookOpen` paths and `Open book` title.
- [ ] Remove assertions for the obsolete `isTitleOnly` branch.
- [ ] Run `npm test`; expect failure against the current title-only data and `48 x 48` A*STAR icon.

## Task 2: Implement The Standard Cards

- [ ] Replace `public/logos/astar.png` with the exact user-supplied official PNG.
- [ ] Populate `content/experience.toml` with the exact approved fields and strings.
- [ ] Simplify `src/components/pages/CardPage.tsx` back to the standard card classes by removing `isTitleOnly` and its conditional sizing.
- [ ] Create `public/favicon-book.svg`, update the configured favicon path, and delete the obsolete heart favicon.
- [ ] Run `npm test`; expect all tests to pass.
- [ ] Commit as `feat: expand experience card details`.

## Task 3: Verify And Deploy

- [ ] Run `npm run check`, `npm run lint`, and `git diff --check`.
- [ ] Verify `1280 x 720` and `390 x 844` browser layouts, loaded logos, exact metadata, the book favicon link, and zero horizontal overflow.
- [ ] Push `master`, wait for the deployment workflow, and verify the uncached live page.
