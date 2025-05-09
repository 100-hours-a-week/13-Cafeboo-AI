import axios from 'axios';

const BASE_URL = 'http://localhost:8000';  // 백엔드 서버 주소

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  },
  withCredentials: false,
});

export interface CaffeineProfile {
  gender: number;
  age: number;
  height: number;
  weight: number;
  is_smoker: number;
  take_hormonal_contraceptive: number;
  caffeine_sensitivity: number;
  current_caffeine: number;
  caffeine_limit: number;
  residual_at_sleep: number;
  target_residual_at_sleep: number;
  planned_caffeine_intake: number;
  current_time: number;
  sleep_time: number;
}

export const getCaffeineLimit = async (profile: CaffeineProfile) => {
  const response = await api.post('/caffeine-limit', profile);
  return response.data;
};

export const getCoffeeRecommendation = async (profile: CaffeineProfile) => {
  const response = await api.post('/coffee-recommendation', profile);
  return response.data;
};

export default api; 