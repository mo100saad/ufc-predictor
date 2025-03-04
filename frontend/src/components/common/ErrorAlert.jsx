import React from 'react';

const ErrorAlert = ({ message }) => {
  return (
    <div className="bg-red-900/30 border border-red-800 p-4 rounded mb-6">
      <p className="text-red-400">{message}</p>
    </div>
  );
};

export default ErrorAlert;