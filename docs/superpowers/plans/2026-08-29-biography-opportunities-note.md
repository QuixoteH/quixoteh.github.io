# Biography Opportunities Note Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the approved research collaboration and internship sentence beneath Biography using the reference site's blockquote format.

**Architecture:** Reuse the existing Markdown source and `ReactMarkdown` blockquote renderer. No component or styling changes are required.

**Tech Stack:** Markdown, Next.js 15, Node.js test runner, static export

## Global Constraints

- The existing Biography paragraph must remain unchanged.
- The sentence must match the approved wording exactly.
- No unrelated content, layout, or styling changes are allowed.

---

### Task 1: Add and verify the Biography blockquote

**Files:**
- Modify: `content/bio.md`
- Modify: `tests/homepage-refresh.test.mjs`
- Modify: `tests/baseline-content.json`

**Interfaces:**
- Consumes: the existing Markdown blockquote renderer in `src/components/home/About.tsx`
- Produces: one blockquote beneath the Biography paragraph on `/`

- [ ] **Step 1: Add a failing source assertion**

Require `content/bio.md` to end with the exact approved Markdown blockquote and add the visible sentence to the homepage export baseline.

- [ ] **Step 2: Verify the assertion fails**

Run `npm test`. Expected: the Biography opportunities assertion fails because the blockquote is absent.

- [ ] **Step 3: Add the approved sentence**

Append exactly:

```markdown
> I am currently looking for research collaboration and internship opportunities related to Force-aware Robot Learning, Robot Manipulation, and Vision-Language-Action Models.
```

- [ ] **Step 4: Verify the site**

Run `npm run check`, `npm run lint`, and `git diff --check`. Expected: all checks pass.

- [ ] **Step 5: Deploy and inspect**

Commit, push `master`, wait for GitHub Pages, and verify desktop and mobile rendering, exact text, zero horizontal overflow, and zero console errors.

