# PRISM Website Migration Design

## Goal

Replace the current Jekyll/AcademicPages implementation with the PRISM Next.js template while preserving the public site's current text content exactly as it appears on GitHub `master` at commit `60aaebf`.

The migration is visual and architectural. It must not rewrite, shorten, expand, translate, or otherwise editorialize the site's biography, interests, news, research, publication, award, or CV content.

## Source Of Truth

- Content baseline: `QuixoteH/quixoteh.github.io` commit `60aaebf`.
- Template baseline: `xyjoey/PRISM` commit `f2748db`.
- Visual reference: the PRISM demo and `gentlefress.github.io`.
- Existing profile image and linked public assets remain the user's assets.

If a difference appears between local historical branches and the baseline commit, the baseline commit wins.

## Architecture

The site will become a statically exported Next.js application using PRISM's existing TypeScript, React, Tailwind CSS, TOML, Markdown, and BibTeX structure.

- `content/` owns all migrated text content and page configuration.
- `public/` owns the profile image, publication media, project images, and downloadable files used by the site.
- `src/` retains PRISM's layout, responsive components, theme handling, and dynamic page rendering.
- A GitHub Actions workflow installs dependencies, runs the production build, and deploys the generated static export to GitHub Pages.

The old Jekyll implementation will be removed after every retained page and asset has a verified PRISM destination. Historical Git data remains available in the repository history.

## Information Architecture

The public navigation labels and URLs remain unchanged:

| Label | URL | PRISM representation |
| --- | --- | --- |
| Home | `/` | About page with profile, biography, interests, selected publication, and news |
| Research | `/portfolio/` | Card page built from the existing portfolio entries |
| Publications | `/publications/` | Publication page backed by BibTeX |
| Awards | `/teaching/` | Card page containing the existing Kaggle award |
| CV | `/cv-json/` | Text page containing the existing structured CV content |

The old `/about/` and `/about.html` entry points will redirect to `/`. Existing collection item URLs will remain available where feasible; otherwise a static redirect page will send visitors to the corresponding retained section.

## Content Preservation

The following fields must remain textually identical to the baseline, except for format-only transformations required by TOML, Markdown, or BibTeX escaping:

- Biography paragraph.
- Three research interests.
- Seven news entries and their dates.
- Research project titles, dates, summaries, and bullet points.
- MSM-Seg title, authorship, venue/status, abstract/description, and public links.
- Kaggle award title, date, description, and ranking.
- CV section labels, entries, dates, descriptions, skills, and languages.
- Public profile name, location, institution, GitHub, and LinkedIn values.

Reusing the existing MSM-Seg publication on the homepage as PRISM's selected publication changes presentation only; it does not add new content.

## Visual Behavior

- Use PRISM's restrained academic layout: fixed top navigation, profile column, serif section headings, compact publication cards, and responsive stacking on small screens.
- Use the current profile image without regeneration or retouching.
- Keep light and dark themes with a visible theme toggle.
- Disable internationalization, the language switcher, and the like button.
- Remove all PRISM example names, publications, institutions, awards, services, and teaching entries.
- Keep the interface in English because the source content and current navigation are English.
- Keep layout dimensions stable so profile content, cards, dates, and navigation do not overlap or shift when content loads.

## Deployment

The repository will store maintainable PRISM source rather than only generated output. GitHub Actions will:

1. Check out the repository.
2. Install the declared Node.js version and dependencies using `npm ci`.
3. Run the production build.
4. Upload the static `out/` directory as the Pages artifact.
5. Deploy the artifact to GitHub Pages.

GitHub Pages will use GitHub Actions as its deployment source. The site remains available at `https://quixoteh.github.io/`.

## Verification

Migration is complete only when all of the following pass from a clean checkout:

- Dependency installation completes from the committed lockfile.
- TypeScript/Next.js production build exits successfully.
- Static export contains `/`, `/portfolio/`, `/publications/`, `/teaching/`, and `/cv-json/`.
- A content contract confirms that all baseline text and required URLs appear in the generated pages.
- No PRISM example content remains in source or generated HTML.
- Desktop and mobile browser checks cover every navigation page in light and dark themes.
- Navigation, profile image, publication links, project media, and theme toggle work.
- Browser console contains no errors or warnings caused by the site.
- `git diff --check` passes and the working tree is clean after commit.
- The pushed GitHub Actions deployment succeeds and the public URL serves the migrated design.

## Scope Boundaries

- No copy editing or factual updates.
- No new bilingual content.
- No like button, analytics, contact form, blog, teaching page, or services page.
- No replacement portrait or generated artwork.
- No unrelated cleanup outside files replaced or made obsolete by the migration.

