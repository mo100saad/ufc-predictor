import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { fighterService } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import { motion } from 'framer-motion';

const Home = () => {
  // Move state variables to the main component
  const [topFighters, setTopFighters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Calculate win percentage with minimum fight requirement
  const calculateWinPercentage = (fighter) => {
    const wins = fighter.wins || 0;
    const losses = fighter.losses || 0;
    const totalFights = wins + losses;
    
    // Require at least 5 fights for consideration
    if (totalFights < 18) return 0;
    return wins / totalFights;
  };
  
  useEffect(() => {
    const loadFighters = async () => {
      try {
        setLoading(true);
        const fighters = await fighterService.getAllFighters();
        
        // Add win percentage to each fighter
        const fightersWithWinPct = fighters.map(fighter => ({
          ...fighter,
          winPct: calculateWinPercentage(fighter)
        }))
        // Filter out fighters with < 80% win rate
        .filter(fighter => fighter.winPct >= 0.8)
        // Sort by win percentage (highest first)
        .sort((a, b) => b.winPct - a.winPct)
        // Take top 5
        .slice(0, 5);
        
        setTopFighters(fightersWithWinPct);
        setLoading(false);
      } catch (err) {
        console.error("Failed to load fighters:", err);
        setError("Failed to load fighters. Please try again later.");
        setLoading(false);
      }
    };
    
    loadFighters();
  }, []);

  // Framer Motion animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      transition={{ duration: 0.5 }}
      className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900"
    >
      {/* Hero Section with Refined Design */}
      <div className="relative min-h-screen flex flex-col">
        {/* Navbar Placeholder - Add actual navbar here with pt-4 or pt-6 */}
        <div className="relative z-20 pt-6">
          {/* Your Navbar Component Would Go Here */}
        </div>

        {/* Background Overlay */}
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-20" 
          style={{ 
            backgroundImage: "url('/path-to-your-hero-image.jpg')",
            filter: 'grayscale(50%) brightness(30%)'
          }}
        ></div>

        {/* Content Container */}
        <div className="relative z-10 container mx-auto px-4 flex-grow flex flex-col">
          {/* Hero Content */}
          <div className="flex-grow flex flex-col justify-center">
            <div className="text-center">
              {/* Refined Title with Elegant Typography */}
              <h1 
                className="text-6xl md:text-8xl font-bebas-neue tracking-wide uppercase text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-red-600 to-red-700 mb-6 drop-shadow-lg"
                style={{ 
                  WebkitTextStroke: '1px rgba(255,255,255,0.2)', 
                }}
              >
                UFC Predictor
              </h1>

              {/* Subtitle with Clear Visibility */}
              <p className="text-lg md:text-xl max-w-2xl mx-auto text-gray-200 mb-10 leading-relaxed font-light opacity-100">
                Harness the power of advanced machine learning to predict UFC fight outcomes with precision. Analyze fighter statistics and get accurate predictions for upcoming matches.
              </p>

              {/* Action Buttons */}
              <motion.div 
                className="flex justify-center space-x-6 mb-16"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.5 }}
              >
                <Link 
                  to="/predict" 
                  className="px-8 py-3 bg-red-600 hover:bg-red-700 text-white rounded-full transform hover:scale-105 transition duration-300 shadow-lg uppercase tracking-wider"
                >
                  Predict a Fight
                </Link>
                <Link 
                  to="/fighters" 
                  className="px-8 py-3 border-2 border-white text-white hover:bg-white hover:text-black rounded-full transform hover:scale-105 transition duration-300 shadow-lg uppercase tracking-wider"
                >
                  Browse Fighters
                </Link>
              </motion.div>
            </div>
          </div>

          {/* Content Grid - Restructured as two tables stacked */}
          <div className="flex flex-col space-y-8 mb-16">
            {/* Top Fighters Section - Full width */}
            <motion.div 
              className="bg-gray-800/60 border border-gray-700 rounded-xl p-6 shadow-2xl backdrop-blur-sm"
              variants={containerVariants}
              initial="hidden"
              animate="show"
            >
              <motion.h2 
                className="text-2xl font-bold mb-6 text-white border-b border-gray-700 pb-3 text-center"
                variants={itemVariants}
              >
                Top Fighters
              </motion.h2>

              {loading ? (
                <LoadingSpinner text="Loading fighters..." />
              ) : error ? (
                <ErrorAlert message={error} />
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  {topFighters.map((fighter, index) => (
                    <motion.div 
                      key={index}
                      variants={itemVariants}
                      whileHover={{ scale: 1.03, transition: { duration: 0.2 } }}
                      className="bg-gradient-to-br from-gray-700/50 to-gray-800/60 rounded-lg overflow-hidden border border-gray-600/30 shadow-lg hover:shadow-xl transition-all duration-300"
                    >
                      <Link to={`/fighters/${encodeURIComponent(fighter.name)}`} className="block">
                        {/* Accent bar at top */}
                        <div className="h-1 bg-gradient-to-r from-red-500 to-red-700"></div>
                        
                        <div className="p-4">
                          {/* Fighter name with icon */}
                          <div className="flex items-center justify-between mb-3">
                            <h3 className="text-lg font-bold text-white">{fighter.name}</h3>
                            <div className="bg-gray-900/50 p-1 rounded-full">
                              <svg className="h-4 w-4 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                              </svg>
                            </div>
                          </div>
                          
                          {/* Stats grid with enhanced styling */}
                          <div className="grid grid-cols-2 gap-2">
                            <div className="bg-gray-900/70 p-2 rounded-lg relative overflow-hidden group">
                              {/* Background accent */}
                              <div className="absolute inset-0 bg-blue-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                              
                              <p className="text-gray-400 text-xs font-medium uppercase tracking-wide">Record</p>
                              <p className="text-white text-lg font-bold">{fighter.wins || 0}-{fighter.losses || 0}</p>
                            </div>
                            
                            <div className="bg-gray-900/70 p-2 rounded-lg relative overflow-hidden group">
                              {/* Background accent */}
                              <div className="absolute inset-0 bg-green-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                              
                              <p className="text-gray-400 text-xs font-medium uppercase tracking-wide">Win Rate</p>
                              <p className="text-gradient bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-green-600 text-lg font-bold">
                                {(fighter.winPct * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                          
                          {/* View details button with hover effect */}
                          <div className="mt-3 text-center">
                            <div className="text-xs text-gray-400 group-hover:text-gray-300 transition-colors duration-300 flex items-center justify-center opacity-70 hover:opacity-100">
                              <span>View Fighter Profile</span>
                              <svg className="ml-1 h-3 w-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                                <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
                              </svg>
                            </div>
                          </div>
                        </div>
                      </Link>
                    </motion.div>
                  ))}
                </div>
              )}
            </motion.div>

            {/* Statistics Section - Full width */}
            <motion.div 
              className="bg-gray-800/60 border border-gray-700 rounded-xl p-6 shadow-2xl backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.5 }}
            >
              <h2 className="text-2xl font-bold mb-6 text-white border-b border-gray-700 pb-3 text-center">
                Prediction Statistics
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gray-700/50 rounded-lg p-6 text-center flex flex-col justify-center items-center border border-gray-600/30 shadow-md h-[120px]">
                  <div className="text-4xl font-bold text-red-500 mb-2">70-90%</div>
                  <div className="text-sm text-gray-300">
                    Prediction Accuracy for Championship Fights
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-6 text-center flex flex-col justify-center items-center border border-gray-600/30 shadow-md h-[120px]">
                  <div className="text-4xl font-bold text-blue-500 mb-2">2,450+</div>
                  <div className="text-sm text-gray-300">
                    UFC Fighters in Database
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-6 text-center flex flex-col justify-center items-center border border-gray-600/30 shadow-md h-[120px]">
                  <div className="text-4xl font-bold text-purple-500 mb-2">12+</div>
                  <div className="text-sm text-gray-300">
                    Fight Statistics Analyzed Per Prediction
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Home;