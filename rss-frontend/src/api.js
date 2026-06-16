import axios from 'axios';

const API = axios.create({
    baseURL: process.env.REACT_APP_API_BASE_URL || 
             (process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8000/api'),
});

export const fetchArticles = (limit = 20) => API.get(`/articles?limit=${limit}`);
export const fetchSummary = (id) => API.get(`/article/${id}/summary`);
export const fetchFeeds = () => API.get('/feeds');
export const addFeed = (feed) => API.post('/feeds', feed);
export const deleteFeed = (id) => API.delete(`/feeds/${id}`);
export const didYouKnowContent = (url) => API.post('/convert-url', {url});
export const processFeed = (id) => API.post(`/process-feed/${id}`);
export const deleteAllArticles = () => API.delete('/articles');


