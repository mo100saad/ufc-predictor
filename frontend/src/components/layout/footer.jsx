import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-gray-900 py-6">
      <div className="container mx-auto px-4 text-center">
        <p className="text-gray-400">
          UFC Fight Predictor &copy; {new Date().getFullYear()} - Powered by Machine Learning
        </p>
        <div className="mt-2">
          <a href="https://github.com/mo100saad/ufc-predictor" 
             className="text-ufc-blue hover:text-ufc-red transition"
             target="_blank"
             rel="noopener noreferrer">
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;