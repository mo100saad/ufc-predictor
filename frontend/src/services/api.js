import axios from 'axios';

const API_URL = 'http://3.138.120.85:5000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fighterService = {
  getAllFighters: async () => {
    const response = await api.get('/fighters');
    return response.data.fighters;
  },
  
  getFighterByName: async (name) => {
    try {
      console.log(`Requesting fighter: ${name}`);
      const response = await api.get(`/fighters/${encodeURIComponent(name)}`);
      console.log('API Response:', response.data);
      return response.data.fighter;
    } catch (error) {
      console.error('Error fetching fighter:', error.response?.data || error.message);
      throw error;
    }
  },
};

export const fightService = {
  getAllFights: async () => {
    const response = await api.get('/fights');
    return response.data.fights;
  },
};

export const predictionService = {
  predictFight: async (fighter1, fighter2) => {
    console.log("Sending prediction request with data:", { fighter1, fighter2 });
    const response = await api.post('/predict', {
      fighter1,
      fighter2
    });
    return response.data.prediction;
  },
};

export const csvService = {
  getCSVStatus: async () => {
    const response = await api.get('/manage-csv');
    return response.data;
  },
  
  syncCSV: async () => {
    const response = await api.post('/manage-csv', {
      action: 'sync'
    });
    return response.data;
  },
};

export default api;
