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
            <h2 className="mb-4 font-serif text-2xl font-bold text-primary">{resolvedTitle}</h2>
            <div className="max-h-80 overflow-y-auto overscroll-y-contain scroll-smooth rounded-lg border border-neutral-200/90 bg-neutral-50/60 px-4 py-3 pr-2 sm:max-h-96 dark:border-neutral-700/80 dark:bg-neutral-900/30">
              <ul className="space-y-4 pr-1">
                {items.map((item, index) => (
                    <li
                        key={`${item.date}-${index}`}
                        className="flex flex-col gap-1 border-b border-neutral-200/60 pb-4 last:border-b-0 last:pb-0 sm:flex-row sm:items-start sm:gap-4 dark:border-neutral-700/50"
                    >
                        <span className="text-sm font-medium tabular-nums text-neutral-500 sm:w-24 sm:flex-shrink-0 sm:text-base">{item.date}</span>
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
