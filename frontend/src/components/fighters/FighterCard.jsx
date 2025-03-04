import React from 'react';
import { Link } from 'react-router-dom';

const FighterCard = ({ fighter }) => {
  return (
    <Link to={`/fighters/${encodeURIComponent(fighter.name)}`}>
      <div className="bg-card-bg hover:bg-gray-800 transition rounded-lg shadow-md p-6 h-full">
        <h2 className="text-xl font-bold mb-2">{fighter.name}</h2>
        <div className="grid grid-cols-2 gap-2 text-sm text-gray-300">
          <div>
            <p><span className="text-gray-500">Record:</span> {fighter.wins}-{fighter.losses}-{fighter.draws || 0}</p>
            <p><span className="text-gray-500">Height:</span> {fighter.height ? `${fighter.height} cm` : 'N/A'}</p>
          </div>
          <div>
            <p><span className="text-gray-500">Weight:</span> {fighter.weight ? `${fighter.weight} lbs` : 'N/A'}</p>
            <p><span className="text-gray-500">Stance:</span> {fighter.stance || 'N/A'}</p>
          </div>
        </div>
      </div>
    </Link>
  );
};

export default FighterCard;