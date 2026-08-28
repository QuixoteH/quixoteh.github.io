'use client';

import Image from 'next/image';
import { motion } from 'framer-motion';
import { Github, Linkedin, MapPin } from 'lucide-react';
import type { SiteConfig } from '@/lib/config';

interface ProfileProps {
  author: SiteConfig['author'];
  social: SiteConfig['social'];
  researchInterests?: string[];
}

export default function Profile({ author, social, researchInterests }: ProfileProps) {
  const socialLinks = [
    ...(social.github ? [{ name: 'GitHub', href: social.github, icon: Github }] : []),
    ...(social.linkedin ? [{ name: 'LinkedIn', href: social.linkedin, icon: Linkedin }] : []),
  ];

  return (
    <motion.aside
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className="lg:sticky lg:top-28"
    >
      <div className="mx-auto mb-6 aspect-square w-full max-w-64 overflow-hidden rounded-lg border border-neutral-200 bg-neutral-100 shadow-md dark:border-neutral-700 dark:bg-neutral-800">
        <Image
          src={author.avatar}
          alt={author.name}
          width={256}
          height={256}
          className="h-full w-full object-cover"
          priority
        />
      </div>

      <div className="mb-5 text-center">
        <h1 className="mb-2 font-serif text-3xl font-bold text-primary">
          {author.name}
        </h1>
        {author.title && (
          <p className="mb-1 text-lg font-medium text-accent">{author.title}</p>
        )}
        <p className="text-neutral-600 dark:text-neutral-400">
          {author.institution}
        </p>
        {social.location && (
          <p className="mt-2 inline-flex items-center gap-1.5 text-sm text-neutral-500 dark:text-neutral-400">
            <MapPin className="h-4 w-4" aria-hidden="true" />
            {social.location}
          </p>
        )}
      </div>

      <div className="mb-6 flex justify-center gap-2">
        {socialLinks.map(({ name, href, icon: Icon }) => (
          <a
            key={name}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-neutral-200 text-neutral-600 transition-colors hover:border-accent hover:text-accent dark:border-neutral-700 dark:text-neutral-400"
            aria-label={name}
            title={name}
          >
            <Icon className="h-5 w-5" aria-hidden="true" />
            <span className="sr-only">{name}</span>
          </a>
        ))}
      </div>

      {researchInterests && researchInterests.length > 0 && (
        <section className="rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-700 dark:bg-neutral-800">
          <h2 className="mb-3 font-semibold text-primary">Interests</h2>
          <ul className="space-y-2 text-sm text-neutral-700 dark:text-neutral-300">
            {researchInterests.map((interest) => (
              <li key={interest}>{interest}</li>
            ))}
          </ul>
        </section>
      )}
    </motion.aside>
  );
}
