import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fighterService } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import FighterImage from '../components/fighters/FighterImage';
import { motion } from 'framer-motion';
import { Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend } from 'chart.js';
import { Radar } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

// Helper functions for formatting
const formatPercentage = (value) => {
  if (value === null || value === undefined) return 'N/A';
  return `${(value * 100).toFixed(0)}%`;
};

const formatRate = (value, suffix) => {
  if (value === null || value === undefined) return 'N/A';
  return `${parseFloat(value).toFixed(1)}${suffix}`;
};

const FighterDetail = () => {
  const { name } = useParams();
  const [fighter, setFighter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const loadFighter = async () => {
      try {
        setLoading(true);
        console.log(`Attempting to load fighter: ${name}`);
        
        const data = await fighterService.getFighterByName(name);
        
        console.log('Received fighter data:', data);
        
        if (!data) {
          console.error(`No data found for fighter: ${name}`);
          setError(`No data found for fighter ${name}`);
          setLoading(false);
          return;
        }
        
        // Ensure we have consistent data
        const cleanedData = {
          ...data,
          SLpM: data.SLpM || 0,
          SApM: data.SApM || 0,
          str_def: data.str_def || 0,
          sig_str_acc: data.sig_str_acc || 0,
          td_avg: data.td_avg || 0,
          td_acc: data.td_acc || 0,
          td_def: data.td_def || 0,
          sub_avg: data.sub_avg || 0,
          wins: data.wins || 0,
          losses: data.losses || 0,
          draws: data.draws || 0
        };
        
        setFighter(cleanedData);
        setLoading(false);
      } catch (err) {
        console.error('Complete error:', err);
        
        // Check if it's a 404 error
        if (err.response && err.response.status === 404) {
          setError(`Fighter not found: ${name}`);
        } else {
          setError(`Failed to load data for ${name}. ${err.message}`);
        }
        
        setLoading(false);
      }
    };
  
    loadFighter();
  }, [name]);

  // Animation variants
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  // Create radar chart data
  const getRadarData = (fighter) => {
    if (!fighter) return null;

    // Normalize values to 0-100 scale
    const normalize = (value, max) => Math.min(100, Math.max(0, (value / max) * 100));

    return {
      labels: ['Striking', 'Takedowns', 'Submissions', 'Defense', 'Experience', 'Power'],
      datasets: [
        {
          label: 'Fighter Stats',
          data: [
            normalize(fighter.SLpM || 0, 8),                // Striking (max ~8 strikes per min)
            normalize(fighter.td_avg || 0, 6),              // Takedowns (max ~6 per fight)
            normalize(fighter.sub_avg || 0, 2),             // Submissions (max ~2 per fight)
            normalize((fighter.str_def || 0.5) * 100, 100), // Defense (percentage)
            normalize(fighter.wins || 0, 30),               // Experience (max ~30 wins)
            normalize((fighter.SLpM || 0) * (fighter.sig_str_acc || 0.5), 4)  // Power - combination of strikes and accuracy
          ],
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 2,
          pointBackgroundColor: 'rgba(255, 99, 132, 1)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgba(255, 99, 132, 1)',
          pointRadius: 4
        }
      ]
    };
  };

  const radarOptions = {
    scales: {
      r: {
        angleLines: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        pointLabels: {
          color: 'rgba(255, 255, 255, 0.7)',
          font: {
            size: 12
          }
        },
        ticks: {
          backdropColor: 'transparent',
          color: 'rgba(255, 255, 255, 0.5)',
          showLabelBackdrop: false
        },
        suggestedMin: 0,
        suggestedMax: 100
      }
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'rgba(255, 99, 132, 1)',
        bodyColor: '#fff',
        cornerRadius: 6,
        padding: 10
      }
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner text={`Loading data for ${name}...`} />
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <ErrorAlert message={error} />
        <Link to="/fighters" className="text-blue-500 hover:text-red-500 transition-colors duration-300 mt-4 inline-block">
          ← Back to fighters
        </Link>
      </div>
    );
  }

  if (!fighter) {
    return (
      <div className="text-center py-10">
        <p className="text-gray-400">Fighter not found</p>
        <Link to="/fighters" className="text-blue-500 hover:text-red-500 transition-colors duration-300 mt-4 inline-block">
          ← Back to fighters
        </Link>
      </div>
    );
  }

  // Calculate fight style based on stats
  const getFightStyle = () => {
    if (!fighter) return "Unknown";
    
    const strikeRatio = (fighter.SLpM || 0) / 5; // Normalized to 0-1 where 5 is high
    const grapplingRatio = ((fighter.td_avg || 0) + (fighter.sub_avg || 0)) / 6; // Normalized
    
    if (strikeRatio > 0.7 && grapplingRatio < 0.3) return "Pure Striker";
    if (strikeRatio < 0.3 && grapplingRatio > 0.7) return "Dominant Grappler";
    if (strikeRatio > 0.5 && grapplingRatio > 0.5) return "Well-Rounded";
    if (strikeRatio > 0.5) return "Striker";
    if (grapplingRatio > 0.5) return "Grappler";
    return "Balanced Fighter";
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="container mx-auto px-4 pt-8"
    >
      <div className="mb-8">
        <Link to="/fighters" className="text-blue-500 hover:text-red-500 transition-colors duration-300 flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to fighters
        </Link>
      </div>
      
      <motion.div 
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 md:grid-cols-3 gap-8"
      >
        {/* Fighter Profile - 1 column */}
        <motion.div variants={item} className="md:col-span-1">
          <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl rounded-lg overflow-hidden">
            {/* Fighter Image Header */}
            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 border-b border-gray-700 flex justify-center">
              <div className="mb-4">
                <FighterImage 
                  src={fighter.image_url} 
                  alt={fighter.name}
                  size="xl" 
                  withBorder={true}
                  borderColor="border-gray-600"
                  className="mx-auto hover:scale-105 transition-transform duration-300"
                />
              </div>
            </div>

            {/* Fighter Name */}
            <div className="px-6 pt-4 pb-2 text-center">
              <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-red-400 to-blue-400">{fighter.name}</h1>
              <div className="text-lg text-gray-300 mt-2 flex items-center justify-center">
                <span className="bg-gray-800 px-2 py-1 rounded text-sm font-medium mr-2">
                  {fighter.wins || 0}-{fighter.losses || 0}{fighter.draws ? `-${fighter.draws}` : ''}
                </span>
                <span className="text-gray-500 text-sm">{getFightStyle()}</span>
              </div>
            </div>
            
            {/* Physical Attributes */}
            <div className="p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-200">Physical Attributes</h2>
              <div className="space-y-3">
                {[
                  { icon: '📏', label: 'Height', value: fighter.height ? `${fighter.height} cm` : 'N/A' },
                  { icon: '⚖️', label: 'Weight', value: fighter.weight ? `${fighter.weight} kg` : 'N/A' },
                  { icon: '🤜', label: 'Reach', value: fighter.reach ? `${fighter.reach} cm` : 'N/A' },
                  { icon: '🥋', label: 'Stance', value: fighter.stance || 'N/A' }
                ].map((attr, index) => (
                  <div key={index} className="flex items-center justify-between bg-gray-800/50 p-3 rounded-lg">
                    <div className="flex items-center">
                      <span className="mr-3">{attr.icon}</span>
                      <span className="text-gray-400">{attr.label}</span>
                    </div>
                    <span className="font-medium text-white">{attr.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Record Card */}
          <motion.div variants={item} className="mt-8">
            <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-200">Professional Record</h2>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-gray-800/50 p-4 rounded-lg text-center">
                  <div className="text-3xl font-bold text-green-500 mb-1">{fighter.wins || 0}</div>
                  <div className="text-xs text-gray-400">Wins</div>
                </div>
                <div className="bg-gray-800/50 p-4 rounded-lg text-center">
                  <div className="text-3xl font-bold text-red-500 mb-1">{fighter.losses || 0}</div>
                  <div className="text-xs text-gray-400">Losses</div>
                </div>
                <div className="bg-gray-800/50 p-4 rounded-lg text-center">
                  <div className="text-3xl font-bold text-yellow-500 mb-1">{fighter.draws || 0}</div>
                  <div className="text-xs text-gray-400">Draws</div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
        
        {/* Stats & Visualization - 2 columns */}
        <motion.div variants={item} className="md:col-span-2">
          <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl rounded-lg p-6">
            <h2 className="text-xl font-bold mb-6 text-gray-200">Performance Analysis</h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Stats - 1 column */}
              <div>
                <h3 className="text-lg font-medium mb-4 text-gray-300">Fighting Statistics</h3>
                <div className="space-y-4">
                  {[
                    { label: 'Striking Rate', value: formatRate(fighter.SLpM, '/min'), bg: 'bg-red-900/20' },
                    { label: 'Striking Accuracy', value: formatPercentage(fighter.sig_str_acc), bg: 'bg-red-900/20' },
                    { label: 'Striking Defense', value: formatPercentage(fighter.str_def), bg: 'bg-blue-900/20' },
                    { label: 'Takedown Average', value: formatRate(fighter.td_avg, '/fight'), bg: 'bg-green-900/20' },
                    { label: 'Takedown Defense', value: formatPercentage(fighter.td_def), bg: 'bg-green-900/20' },
                    { label: 'Submission Average', value: formatRate(fighter.sub_avg, '/fight'), bg: 'bg-purple-900/20' }
                  ].map((stat, index) => (
                    <div key={index} className={`flex justify-between items-center p-3 rounded-lg ${stat.bg}`}>
                      <span className="text-gray-300">{stat.label}</span>
                      <span className="font-medium text-white">{stat.value}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Radar Chart - 1 column */}
              <div className="flex flex-col justify-center">
                <h3 className="text-lg font-medium mb-4 text-gray-300 text-center">Fighter Radar</h3>
                <div className="h-64">
                  {getRadarData(fighter) && (
                    <Radar data={getRadarData(fighter)} options={radarOptions} />
                  )}
                </div>
              </div>
            </div>
            
            {/* Fighter strength analysis */}
            <div className="mt-8">
              <h3 className="text-lg font-medium mb-4 text-gray-300">Fighter Analysis</h3>
              <div className="bg-gray-800/50 p-5 rounded-lg">
                <div className="space-y-4">
                  {/* Fighter Profile Overview */}
                  <div className="flex flex-col">
                    <div className="flex items-center">
                      <div className="h-2 w-2 bg-red-500 rounded-full mr-2"></div>
                      <span className="text-lg font-semibold text-gray-200">
                        {fighter.name} is a {getFightStyle().toLowerCase()} with {fighter.wins || 0} professional wins
                      </span>
                    </div>
                  </div>
                  
                  {/* Fighter Key Strengths - only show if values exist */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {fighter.SLpM > 4 && (
                      <div className="bg-gray-700/30 p-3 rounded-lg">
                        <div className="flex items-center">
                          <div className="mr-2 text-red-400">⚡</div>
                          <span className="text-gray-300">High striking output: <span className="font-medium text-white">{fighter.SLpM.toFixed(1)} strikes/min</span></span>
                        </div>
                      </div>
                    )}
                    
                    {fighter.td_avg > 3 && (
                      <div className="bg-gray-700/30 p-3 rounded-lg">
                        <div className="flex items-center">
                          <div className="mr-2 text-blue-400">💪</div>
                          <span className="text-gray-300">Excellent takedowns: <span className="font-medium text-white">{fighter.td_avg.toFixed(1)} per fight</span></span>
                        </div>
                      </div>
                    )}
                    
                    {fighter.sub_avg > 1 && (
                      <div className="bg-gray-700/30 p-3 rounded-lg">
                        <div className="flex items-center">
                          <div className="mr-2 text-green-400">🔒</div>
                          <span className="text-gray-300">Submission specialist: <span className="font-medium text-white">{fighter.sub_avg.toFixed(1)} per fight</span></span>
                        </div>
                      </div>
                    )}
                    
                    {fighter.td_def > 0.8 && (
                      <div className="bg-gray-700/30 p-3 rounded-lg">
                        <div className="flex items-center">
                          <div className="mr-2 text-purple-400">🛡️</div>
                          <span className="text-gray-300">Exceptional takedown defense: <span className="font-medium text-white">{(fighter.td_def * 100).toFixed(0)}%</span></span>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Fighting Style */}
                  <div className="bg-gray-700/30 p-3 rounded-lg">
                    <div className="flex items-center">
                      <div className="mr-2 text-yellow-400">📊</div>
                      <span className="text-gray-300">
                        {fighter.SLpM > (fighter.SApM || 0) ? 
                          `Offensive fighter who lands more strikes (${(fighter.SLpM || 0).toFixed(1)}) than absorbed (${(fighter.SApM || 0).toFixed(1)})` : 
                          `Takes more strikes (${(fighter.SApM || 0).toFixed(1)}) than lands (${(fighter.SLpM || 0).toFixed(1)})`
                        }
                      </span>
                    </div>
                  </div>
                  
                  {/* Record Summary */}
                  <div className="bg-gray-700/30 p-3 rounded-lg">
                    <div className="flex items-center">
                      <div className="mr-2 text-green-400">🏆</div>
                      <span className="text-gray-300">
                        Professional record of {fighter.wins || 0} wins and {fighter.losses || 0} losses
                        {fighter.draws > 0 ? ` with ${fighter.draws} draws` : ''}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Create Prediction Button */}
          <motion.div variants={item} className="mt-8">
            <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-200">Fight Predictions</h2>
              <p className="text-gray-300 mb-6">
                Want to see how {fighter.name} would perform against another fighter? Use our advanced prediction engine to simulate a matchup.
              </p>
              <Link 
                to="/predict" 
                className="block text-center py-3 px-6 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-bold rounded-lg shadow-lg transition-all duration-300 transform hover:scale-105"
              >
                Create a Fight Prediction
              </Link>
            </div>
          </motion.div>
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default FighterDetail;