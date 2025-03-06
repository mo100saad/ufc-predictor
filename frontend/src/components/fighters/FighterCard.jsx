import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const FighterCard = ({ fighter }) => {
  if (!fighter) return null;
  
  // Calculate win percentage
  const totalFights = (fighter.wins || 0) + (fighter.losses || 0);
  const winPercentage = totalFights > 0 ? ((fighter.wins || 0) / totalFights * 100).toFixed(0) : 0;
  
  // Determine fighter style based on stats
  const getFighterStyle = () => {
    const strikingRatio = (fighter.SLpM || 0) / 5; // Normalized to 0-1 where 5 is high
    const grapplingRatio = ((fighter.td_avg || 0) + (fighter.sub_avg || 0)) / 6; // Normalized
    
    if (strikingRatio > 0.7 && grapplingRatio < 0.3) return "Striker";
    if (strikingRatio < 0.3 && grapplingRatio > 0.7) return "Grappler";
    if (strikingRatio > 0.5 && grapplingRatio > 0.5) return "Well-Rounded";
    return "Balanced";
  };
  
  // Determine fighter's weight class
  const getWeightClass = (weight) => {
    if (!weight) return "Unknown";
    
    if (weight <= 125) return "Flyweight";
    if (weight <= 135) return "Bantamweight";
    if (weight <= 145) return "Featherweight";
    if (weight <= 155) return "Lightweight";
    if (weight <= 170) return "Welterweight";
    if (weight <= 185) return "Middleweight";
    if (weight <= 205) return "Light Heavyweight";
    return "Heavyweight";
  };

  return (
    <motion.div 
      className="group h-full"
      whileHover={{ y: -5 }}
      transition={{ duration: 0.2 }}
    >
      <Link 
        to={`/fighter/${encodeURIComponent(fighter.name)}`}
        className="block h-full"
      >
        <div className="relative h-full backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-lg hover:shadow-xl rounded-lg overflow-hidden transition-all duration-300 hover:border-gray-700 group-hover:bg-gray-900/80">
          
          {/* Gradient top accent */}
          <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-red-500 to-blue-500"></div>
          
          {/* Card Content */}
          <div className="p-5">
            {/* Fighter Name */}
            <h3 className="text-xl font-bold text-white mb-1 group-hover:text-blue-400 transition-colors duration-300">{fighter.name}</h3>
            
            {/* Record and Division */}
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center">
                <span className="text-sm font-medium bg-gray-800 text-gray-300 px-2 py-1 rounded">
                  {fighter.wins || 0}-{fighter.losses || 0}{fighter.draws ? `-${fighter.draws}` : ''}
                </span>
                {winPercentage > 0 && (
                  <span className={`ml-2 text-xs px-1.5 py-0.5 rounded ${
                    winPercentage >= 70 ? 'bg-green-900/50 text-green-400' : 
                    winPercentage >= 50 ? 'bg-blue-900/50 text-blue-400' : 
                    'bg-red-900/50 text-red-400'
                  }`}>
                    {winPercentage}% Wins
                  </span>
                )}
              </div>
              <span className="text-xs text-gray-500">
                {getWeightClass(fighter.weight)}
              </span>
            </div>
            
            {/* Fighter Stats - mini grid */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              <div className="text-center p-2 bg-gray-800/50 rounded-lg">
                <div className="text-lg font-bold text-red-500">{fighter.SLpM?.toFixed(1) || 'N/A'}</div>
                <div className="text-xs text-gray-400">Strikes/Min</div>
              </div>
              
              <div className="text-center p-2 bg-gray-800/50 rounded-lg">
                <div className="text-lg font-bold text-blue-500">{fighter.td_avg?.toFixed(1) || 'N/A'}</div>
                <div className="text-xs text-gray-400">Takedowns/Fight</div>
              </div>
            </div>
            
            {/* Fighter Style & Last Row */}
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-400">
                Style: <span className="text-gray-300">{getFighterStyle()}</span>
              </span>
              
              <span className="text-blue-500 text-xs transition-all duration-300 opacity-0 group-hover:opacity-100 transform translate-x-2 group-hover:translate-x-0">
                View Details →
              </span>
            </div>
          </div>
        </div>
      </Link>
    </motion.div>
  );
};

export default FighterCard;