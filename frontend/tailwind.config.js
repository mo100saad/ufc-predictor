/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'ufc-red': '#D20A0A',  // UFC red color
        'ufc-blue': '#1277BC', // UFC blue color
        'ufc-gold': '#D4AF37', // Gold for champions
      },
      fontFamily: {
        sans: ['Poppins', 'sans-serif'],
      },
      animation: {
        'bounce-slow': 'bounce 3s infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      },
      boxShadow: {
        'glow-red': '0 0 15px rgba(220, 38, 38, 0.5)',
        'glow-blue': '0 0 15px rgba(59, 130, 246, 0.5)',
        'glow-white': '0 0 15px rgba(255, 255, 255, 0.5)',
      },
    },
  },
  plugins: [],
}