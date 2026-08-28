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
  assert.equal(config.site.favicon, '/favicon-book.svg');
  assert.equal(config.author.name, 'Hai Huang');
  assert.equal(config.author.title, 'M.S. Student in Robotics');
  assert.equal(config.social.email, 'quixotehh@gmail.com');
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
      { id: 'about', type: 'markdown', source: 'bio.md', title: 'Biography' },
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
      content: 'College of Intelligent Science and Engineering',
      image: '/logos/neau.png',
    },
  ]);

  const experience = readToml('content/experience.toml');
  assert.deepEqual(experience.items, [
    {
      title: 'Research Attachment',
      subtitle: 'Singapore Institute of Manufacturing Technology (SIMTech), A*STAR',
      date: '08/2026 – Present',
      content: 'Singapore',
      image: '/logos/astar.png',
      tags: ['Force-aware Robot Learning', 'Robot Manipulation'],
    },
    {
      title: 'AI Solutions Engineer Intern',
      subtitle: 'China Unicom Chengdu Branch, Digital Technology Center',
      date: '07/2025 – 09/2025',
      content: 'Chengdu, China',
      image: '/logos/china-unicom.png',
      tags: ['AI Solution', 'IoT Solution'],
    },
  ]);
  for (const item of experience.items) {
    assert.deepEqual(Object.keys(item), ['title', 'subtitle', 'date', 'content', 'image', 'tags']);
  }
});

test('profile email and Experience logo assets are wired', () => {
  const profile = read('src/components/home/Profile.tsx');
  assert.match(profile, /Github, Linkedin, Mail, MapPin/);
  assert.match(profile, /name: 'Email'/);
  assert.match(profile, /`mailto:\$\{social\.email\}`/);

  for (const logo of ['public/logos/astar.png', 'public/logos/china-unicom.png']) {
    assert.ok(fs.statSync(path.join(root, logo)).size > 0);
  }

  const astar = fs.readFileSync(path.join(root, 'public/logos/astar.png'));
  assert.deepEqual([astar.readUInt32BE(16), astar.readUInt32BE(20)], [675, 675]);

  const favicon = read('public/favicon-book.svg');
  assert.match(favicon, /<title>Open book<\/title>/);
  assert.match(favicon, /M12 7v14/);
  assert.match(favicon, /M3 18a1 1 0 0 1-1-1V4/);
});

test('homepage loader and renderer support card sections', () => {
  const page = read('src/app/page.tsx');
  const client = read('src/components/home/HomePageClient.tsx');
  assert.match(page, /type: 'markdown' \| 'publications' \| 'list' \| 'card'/);
  assert.match(page, /case 'card'/);
  assert.match(client, /case 'card'/);
  assert.doesNotMatch(client, /SelectedPublications/);
});

test('GentleFress visual structure and light-default theme are configured', () => {
  const card = read('src/components/pages/CardPage.tsx');
  const news = read('src/components/home/News.tsx');
  const store = read('src/lib/stores/themeStore.ts');
  const layout = read('src/app/layout.tsx');
  const css = read('src/app/globals.css');

  assert.match(card, /item\.image/);
  assert.match(card, /h-12 w-12/);
  assert.doesNotMatch(card, /isTitleOnly/);
  assert.match(news, /max-h-80/);
  assert.match(news, /sm:max-h-96/);
  assert.match(news, /overflow-y-auto/);
  assert.match(news, /ReactMarkdown/);
  assert.match(store, /theme: 'light'/);
  assert.match(layout, /parsed\?\.state\?\.theme \|\| 'light'/);
  assert.match(css, /--accent: #7D6B8C;/);
  assert.match(css, /--accent-dark: #675873;/);
});

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
  assert.match(cv, /85\/100/);
});
