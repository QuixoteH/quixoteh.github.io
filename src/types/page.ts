export interface BasePageConfig {
    type: 'about' | 'publication' | 'card' | 'competition' | 'text';
    title: string;
    description?: string;
}

export interface PublicationPageConfig extends BasePageConfig {
    type: 'publication';
    source: string;
}

export interface TextPageConfig extends BasePageConfig {
    type: 'text';
    source: string;
}

export interface CardItem {
    title: string;
    subtitle?: string;
    date?: string;
    content?: string;
    tags?: string[];
    link?: string;
    image?: string;
}

export interface CardPageConfig extends BasePageConfig {
    type: 'card';
    items: CardItem[];
}

export interface CompetitionLink {
    label: string;
    url: string;
    kind?: 'code' | 'competition' | 'paper' | 'video';
}

export interface CompetitionItem {
    title: string;
    badge?: string;
    role?: string;
    organization: string;
    date: string;
    content: string;
    images?: string[];
    visuals?: string[];
    links?: CompetitionLink[];
}

export interface CompetitionPageConfig extends BasePageConfig {
    type: 'competition';
    items: CompetitionItem[];
}
