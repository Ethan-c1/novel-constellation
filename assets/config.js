// 公开版本不内置第三方音乐。可将有权使用的音频放入 music/，
// 再把对应文件名添加到此数组。
export const MUSIC_LIST = [];

function normalizeApiBase(value) {
    return (value || '').trim().replace(/\/+$/, '');
}

export function resolveApiBase() {
    const queryValue = new URLSearchParams(window.location.search).get('api');
    if (queryValue) {
        const normalized = normalizeApiBase(queryValue);
        localStorage.setItem('relationshipApiBase', normalized);
        return normalized;
    }
    const savedValue = localStorage.getItem('relationshipApiBase');
    const metaValue = document.querySelector('meta[name="api-base"]')?.content;
    return normalizeApiBase(
        savedValue
        || metaValue
        || `${window.location.protocol}//${window.location.hostname}:5001`
    );
}
