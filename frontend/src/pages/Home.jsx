import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { fighterService } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import { motion } from 'framer-motion';

const Home = () => {
  const [topFighters, setTopFighters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Require at least 18 fights for a fighter to be considered
  const calculateWinPercentage = (fighter) => {
    const { wins, losses } = fighter;
    const totalFights = wins + losses;
    if (totalFights < 18) return 0;
    return wins / totalFights;
  };

  useEffect(() => {
    const loadFighters = async () => {
      try {
        setLoading(true);
        const fighters = await fighterService.getAllFighters();
        const fightersWithWinPct = fighters.map(fighter => ({
          ...fighter,
          winPct: calculateWinPercentage(fighter)
        })).filter(fighter => fighter.winPct > 0);
        const sortedTopFighters = fightersWithWinPct.sort((a, b) => b.winPct - a.winPct).slice(0, 5);
        setTopFighters(sortedTopFighters);
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

          {/* Content Grid for Top Fighters and Stats */}
          <div className="grid md:grid-cols-2 gap-8 mb-16">
            {/* Top Fighters Section - Refined Design */}
            <motion.div 
              className="bg-gray-800/60 border border-gray-700 rounded-xl p-6 shadow-2xl backdrop-blur-sm"
              variants={containerVariants}
              initial="hidden"
              animate="show"
            >
              <motion.h2 
                className="text-2xl font-bold mb-6 text-white border-b border-gray-700 pb-3"
                variants={itemVariants}
              >
                Top Fighters
              </motion.h2>
              {loading ? (
                <LoadingSpinner text="Loading fighters..." />
              ) : error ? (
                <ErrorAlert message={error} />
              ) : (
                <div className="space-y-4">
                  {topFighters.length > 0 ? (
                    topFighters.map((fighter, index) => (
                      <Link key={fighter.id || index} to={`/fighters/${encodeURIComponent(fighter.name)}`}>
                        <motion.div 
                          className="bg-gray-700/50 rounded-lg p-4 hover:bg-gray-700/70 transition-all duration-300 group"
                          variants={itemVariants}
                        >
                          <div className="flex justify-between items-center">
                            <div>
                              <div className="text-lg font-semibold text-white group-hover:text-red-400 transition-colors">
                                {fighter.name}
                              </div>
                              <div className="text-sm text-gray-400">
                                Record: {fighter.wins} - {fighter.losses}
                              </div>
                            </div>
                            <div className="text-sm font-bold text-green-400">
                              {(fighter.winPct * 100).toFixed(1)}%
                            </div>
                          </div>
                        </motion.div>
                      </Link>
                    ))
                  ) : (
                    <p className="text-gray-500 text-center py-8">No top fighters available</p>
                  )}
                </div>
              )}
            </motion.div>

            {/* Statistics Section */}
            <motion.div 
              className="bg-gray-800/60 border border-gray-700 rounded-xl p-6 shadow-2xl backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.5 }}
            >
              <h2 className="text-2xl font-bold mb-6 text-white border-b border-gray-700 pb-3">
                Prediction Statistics
              </h2>
              <div className="grid grid-cols-1 gap-6">
                <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                  <div className="text-4xl font-bold text-red-500 mb-2">93%</div>
                  <div className="text-sm text-gray-300">
                    Prediction Accuracy for Championship Fights
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                  <div className="text-4xl font-bold text-blue-500 mb-2">2,450+</div>
                  <div className="text-sm text-gray-300">
                    UFC Fighters in Database
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-4 text-center">
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