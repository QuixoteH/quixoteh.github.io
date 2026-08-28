'use client';

import { motion } from 'framer-motion';
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
            <h2 className="text-2xl font-serif font-bold text-primary mb-4">{resolvedTitle}</h2>
            <div className="space-y-3">
                {items.map((item, index) => (
                    <div key={index} className="grid grid-cols-[5.5rem_minmax(0,1fr)] items-start gap-3">
                        <span className="mt-0.5 text-xs text-neutral-500">{item.date}</span>
                        <p className="text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">{item.content}</p>
                    </div>
                ))}
            </div>
        </motion.section>
    );
}
