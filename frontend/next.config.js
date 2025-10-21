/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://aie8.onrender.com',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://aie8.onrender.com/api/:path*',
      },
    ]
  },
}

module.exports = nextConfig


