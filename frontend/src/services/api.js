import axios from 'axios';

// Define API URL with fallback for news API
const API_URL = '/api';
const NEWS_URL = process.env.NODE_ENV === 'development' ? 'http://localhost:5050' : '/api';

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
  
  // Enhanced fighter image service with better fallback sequence
  getFighterImage: async (name) => {
    try {
      // First try to get the image from our API (which tries UFC.com)
      const response = await api.get(`/fighters/${encodeURIComponent(name)}/image`);
      const imageUrl = response.data.image_url;
      
      // If we got a valid image, return it
      if (imageUrl && imageUrl !== "/static/placeholder.png") {
        console.log(`Found image for ${name} from primary source`);
        return imageUrl;
      }
      
      // If we didn't get a valid image, try fallback sources
      console.log(`Primary image source failed for ${name}, trying fallbacks...`);
      
      // Try fallback sequence - implemented on backend, pass source parameter
      const fallbacks = ['wikipedia', 'sherdog'];
      
      // Try each fallback source in sequence
      for (const source of fallbacks) {
        try {
          const fallbackResponse = await api.get(
            `/fighters/${encodeURIComponent(name)}/image?source=${source}`
          );
          const fallbackUrl = fallbackResponse.data.image_url;
          
          // If we got a valid image from fallback, return it
          if (fallbackUrl && fallbackUrl !== "/static/placeholder.png") {
            console.log(`Found image for ${name} from ${source}`);
            return fallbackUrl;
          }
        } catch (fallbackError) {
          console.log(`Failed to get image from ${source} for ${name}`);
        }
      }
      
      // If all fallbacks failed, return placeholder
      return "/static/placeholder.png";
    } catch (error) {
      console.error('Error fetching fighter image:', error.response?.data || error.message);
      return "/static/placeholder.png";
    }
  }
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
    
    // Apply client-side weight class correction for extreme mismatches before sending to API
    const adjustedFighters = applyWeightClassCorrection(fighter1, fighter2);
    
    const response = await api.post('/predict', {
      fighter1: adjustedFighters.fighter1,
      fighter2: adjustedFighters.fighter2
    });
    
    // Get prediction from server
    let prediction = response.data.prediction;
    
    // Apply additional probability adjustment for extreme weight differences
    prediction = adjustPredictionForWeightDisparity(prediction, fighter1, fighter2);
    
    return prediction;
  },
};

// Helper function to apply weight class correction to prevent unrealistic predictions
function applyWeightClassCorrection(fighter1, fighter2) {
  // Deep clone the fighters to avoid modifying originals
  const f1 = JSON.parse(JSON.stringify(fighter1));
  const f2 = JSON.parse(JSON.stringify(fighter2));
  
  // Calculate weight difference
  const weight1 = f1.weight || 0;
  const weight2 = f2.weight || 0;
  const weightDiff = Math.abs(weight1 - weight2);
  
  // If weight difference is extreme (more than 15kg / ~33lbs), apply corrections
  if (weightDiff > 15) {
    console.log(`Extreme weight difference detected: ${weightDiff.toFixed(1)}kg - applying correction`);
    
    // Determine which fighter is heavier
    const heavierFighter = weight1 > weight2 ? f1 : f2;
    const lighterFighter = weight1 > weight2 ? f2 : f1;
    
    // Apply a penalty to the lighter fighter's stats based on the weight difference
    const penaltyFactor = Math.min(0.7, Math.max(0.3, 1 - (weightDiff / 100)));
    
    // Reduce striking metrics for the lighter fighter against a much heavier opponent
    lighterFighter.SLpM = lighterFighter.SLpM * penaltyFactor;
    lighterFighter.str_acc = lighterFighter.str_acc * penaltyFactor;
    
    console.log(`Applied penalty factor ${penaltyFactor.toFixed(2)} to lighter fighter's offensive stats`);
  }
  
  return { fighter1: f1, fighter2: f2 };
}

// Helper function to adjust prediction probabilities for extreme weight disparities
function adjustPredictionForWeightDisparity(prediction, fighter1, fighter2) {
  // Guard against malformed prediction object
  if (!prediction || typeof prediction !== 'object') return prediction;
  
  const weight1 = fighter1.weight || 0;
  const weight2 = fighter2.weight || 0;
  const weightDiff = Math.abs(weight1 - weight2);
  
  // Define weight class difference thresholds
  // 10kg (~22lbs) = approximately 1 weight class
  // 20kg (~44lbs) = approximately 2 weight classes
  // 30kg (~66lbs) = approximately 3 weight classes or more
  
  if (weightDiff > 20) {
    // Clone prediction to avoid modifying original
    const adjustedPrediction = {...prediction};
    
    // Determine which fighter is heavier and adjust probabilities
    if (weight1 > weight2) {
      // Fighter 1 is heavier
      const adjustmentFactor = Math.min(0.3, weightDiff / 100);
      
      // Take probability from lighter fighter and give to heavier fighter
      adjustedPrediction.fighter2_win_probability = Math.max(0.05, 
        adjustedPrediction.fighter2_win_probability - adjustmentFactor);
      adjustedPrediction.fighter1_win_probability = Math.min(0.95, 
        1 - adjustedPrediction.fighter2_win_probability);
      
      console.log(`Adjusted prediction for weight disparity: ${adjustmentFactor.toFixed(2)} in favor of heavier fighter 1`);
    } else {
      // Fighter 2 is heavier
      const adjustmentFactor = Math.min(0.3, weightDiff / 100);
      
      // Take probability from lighter fighter and give to heavier fighter
      adjustedPrediction.fighter1_win_probability = Math.max(0.05, 
        adjustedPrediction.fighter1_win_probability - adjustmentFactor);
      adjustedPrediction.fighter2_win_probability = Math.min(0.95, 
        1 - adjustedPrediction.fighter1_win_probability);
      
      console.log(`Adjusted prediction for weight disparity: ${adjustmentFactor.toFixed(2)} in favor of heavier fighter 2`);
    }
    
    return adjustedPrediction;
  }
  
  return prediction;
}

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

// Create a separate news API instance
const newsApi = axios.create({
  baseURL: NEWS_URL,
  headers: {
    'Content-Type': 'application/json',
  }
});

export const newsService = {
  getUFCNews: async () => {
    try {
      console.log(`Attempting to fetch news from ${NEWS_URL}/news`);
      const response = await newsApi.get('/news');
      console.log("News API response:", response.data);
      return response.data.news || [];
    } catch (error) {
      console.error("Error fetching UFC news:", error);
      // More detailed error logging
      if (error.response) {
        console.error("Response data:", error.response.data);
        console.error("Response status:", error.response.status);
        console.error("Response headers:", error.response.headers);
      } else if (error.request) {
        console.error("Request made but no response received:", error.request);
      } else {
        console.error("Error message:", error.message);
      }
      
      // Try direct NewsAPI call as a last resort
      try {
        console.log("Attempting direct NewsAPI call - this may not work in production due to CORS");
        const encodedQuery = encodeURIComponent('UFC OR "Ultimate Fighting Championship" OR MMA');
        // Note: Direct API calls no longer recommended - using backend proxy is better
        // API key has been removed from client code for security
        const backendProxyUrl = `/api/news`;
        
        const directResponse = await axios.get(backendProxyUrl);
        if (directResponse.data && directResponse.data.news) {
          console.log("Backend news API call successful");
          const articles = directResponse.data.news || [];
          
          // Process articles similar to server-side
          return articles.map(article => ({
            id: article.url,
            title: article.title,
            description: article.description,
            url: article.url,
            imageUrl: article.urlToImage || '/static/placeholder.png',
            source: article.source?.name || 'News Source',
            publishedAt: article.publishedAt || new Date().toISOString()
          }));
        }
      } catch (directError) {
        console.error("Direct NewsAPI call also failed:", directError);
      }
      
      // Fallback to static content as a last resort
      return [
        {
          id: 1,
          title: "UFC News Temporarily Unavailable",
          description: "We're working to restore the latest UFC news feed. Check back soon!",
          url: "https://ufc.com/news",
          imageUrl: "/static/placeholder.png",
          source: "System",
          publishedAt: new Date().toISOString()
        }
      ];
    }
  }
};

export default api;
