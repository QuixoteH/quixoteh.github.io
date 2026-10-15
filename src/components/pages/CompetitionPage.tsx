'use client';

import Image from 'next/image';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { Code2, ExternalLink, FileText, PlayCircle, Trophy } from 'lucide-react';
import type { CompetitionLink, CompetitionPageConfig } from '@/types/page';

const markdownComponents = {
    ul: ({ children }: React.ComponentProps<'ul'>) => <ul className="space-y-2.5">{children}</ul>,
    li: ({ children }: React.ComponentProps<'li'>) => (
        <li className="relative pl-5 leading-relaxed before:absolute before:left-0 before:top-0 before:font-bold before:text-accent before:content-['›']">
            {children}
        </li>
    ),
    strong: ({ children }: React.ComponentProps<'strong'>) => <strong className="font-semibold text-primary">{children}</strong>,
    a: ({ ...props }) => (
        <a {...props} target="_blank" rel="noopener noreferrer" className="font-medium text-accent hover:underline" />
    ),
};

function LinkIcon({ kind }: { kind?: CompetitionLink['kind'] }) {
    const className = 'h-4 w-4';
    if (kind === 'code') return <Code2 className={className} aria-hidden="true" />;
    if (kind === 'video') return <PlayCircle className={className} aria-hidden="true" />;
    if (kind === 'paper') return <FileText className={className} aria-hidden="true" />;
    return <ExternalLink className={className} aria-hidden="true" />;
}

export default function CompetitionPage({ config }: { config: CompetitionPageConfig }) {
    return (
        <motion.div initial={false} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <div className="mb-8">
                <h1 className="mb-4 font-serif text-4xl font-bold text-primary">{config.title}</h1>
                {config.description && (
                    <p className="max-w-2xl text-lg leading-relaxed text-neutral-600 dark:text-neutral-500">
                        {config.description}
                    </p>
                )}
            </div>

            <div className="space-y-8">
                {config.items.map((item, index) => {
                    const visualItems = item.images?.length ? item.images : item.visuals || [];

                    return (
                        <motion.article
                            key={item.title}
                            initial={false}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4, delay: index * 0.08 }}
                            className="overflow-hidden rounded-xl border border-neutral-200 bg-white shadow-sm transition-all duration-200 hover:-translate-y-1 hover:border-accent/40 hover:shadow-md dark:border-neutral-800 dark:bg-neutral-900"
                        >
                            {visualItems.length > 0 && (
                                <div className={`grid h-44 ${visualItems.length > 1 ? 'grid-cols-2' : 'grid-cols-1'} border-b border-neutral-200 dark:border-neutral-800 sm:h-56`}>
                                    {visualItems.slice(0, 2).map((visual, visualIndex) => (
                                        item.images?.length ? (
                                            <div key={visual} className="relative overflow-hidden bg-neutral-100 dark:bg-neutral-800">
                                                <Image
                                                    src={visual}
                                                    alt={`${item.title} preview ${visualIndex + 1}`}
                                                    fill
                                                    unoptimized
                                                    sizes={visualItems.length > 1 ? '(max-width: 896px) 50vw, 448px' : '(max-width: 896px) 100vw, 896px'}
                                                    className="object-cover"
                                                />
                                            </div>
                                        ) : (
                                            <div
                                                key={visual}
                                                className="flex items-center justify-center bg-gradient-to-br from-accent/15 via-accent/5 to-neutral-100 px-5 text-center dark:from-accent/20 dark:via-neutral-900 dark:to-neutral-800"
                                            >
                                                <div>
                                                    <Trophy className="mx-auto mb-3 h-7 w-7 text-accent" aria-hidden="true" />
                                                    <span className="font-serif text-xl font-semibold text-primary sm:text-2xl">{visual}</span>
                                                </div>
                                            </div>
                                        )
                                    ))}
                                </div>
                            )}

                            <div className="p-5 sm:p-6">
                                <div className="mb-3 flex flex-col items-start gap-3 sm:flex-row sm:justify-between">
                                    <h2 className="font-serif text-2xl font-semibold leading-tight text-primary">{item.title}</h2>
                                    {item.badge && (
                                        <span className="w-fit flex-shrink-0 rounded-full border border-accent/30 bg-accent/10 px-3 py-1.5 text-sm font-semibold text-accent">
                                            {item.badge}
                                        </span>
                                    )}
                                </div>

                                {item.role && <p className="mb-2 font-medium text-primary">{item.role}</p>}
                                <p className="mb-5 text-sm font-medium text-neutral-500">
                                    {item.organization} <span aria-hidden="true">·</span> {item.date}
                                </p>

                                <div className="text-base text-neutral-600 dark:text-neutral-400">
                                    <ReactMarkdown components={markdownComponents}>{item.content}</ReactMarkdown>
                                </div>

                                {item.links && item.links.length > 0 && (
                                    <div className="mt-6 flex flex-wrap gap-x-5 gap-y-3 border-t border-neutral-200 pt-4 dark:border-neutral-800">
                                        {item.links.map((link) => (
                                            <a
                                                key={`${link.label}-${link.url}`}
                                                href={link.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="inline-flex items-center gap-1.5 text-sm font-medium text-accent transition-opacity hover:opacity-70"
                                            >
                                                <LinkIcon kind={link.kind} />
                                                {link.label}
                                            </a>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </motion.article>
                    );
                })}
            </div>
        </motion.div>
    );
}
