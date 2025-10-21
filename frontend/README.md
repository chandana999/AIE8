# 🎨 Multi-Agent Log Analyzer - Frontend

Modern Next.js frontend for the Multi-Agent Log Analyzer system, providing an intuitive interface for Apache log analysis with real-time results and professional SRE-focused output.

## 🎯 **Features**

### **Core Functionality**
- ✅ **File Upload Interface** with drag-and-drop support
- ✅ **Real-time Analysis** with streaming responses
- ✅ **Professional UI** with Tailwind CSS styling
- ✅ **API Key Management** for OpenAI and Tavily
- ✅ **Error Handling** with detailed feedback
- ✅ **Responsive Design** for all devices

### **Analysis Display**
- 🧠 **Incident Summary** with severity indicators
- 🕒 **Event Timeline** with chronological flow
- ⚙️ **Root Cause Analysis** with causal chains
- 🚑 **Immediate Remediation** with actionable steps
- 🧱 **Prevention Recommendations** for long-term fixes

### **User Experience**
- 🎨 **Modern Design** with gradient buttons and cards
- 📱 **Mobile Responsive** layout
- ⚡ **Fast Loading** with optimized components
- 🔄 **Real-time Updates** during analysis
- 🎯 **Intuitive Navigation** with clear sections

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js       │    │   FastAPI        │    │   Multi-Agent   │
│   Frontend      │◄──►│   Backend        │◄──►│   System        │
│                 │    │                  │    │                 │
│ • File Upload   │    │ • API Endpoints  │    │ • LogSearch     │
│ • Analysis UI   │    │ • Multi-Agent    │    │ • LogAnalysisRAG│
│ • Real-time     │    │ • Streaming      │    │ • Supervisor    │
│   Results       │    │ • Validation     │    │ • Routing       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 **Getting Started**

### **Prerequisites**
- Node.js 18+
- npm or yarn
- Backend API running (FastAPI server)

### **Installation**

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📋 **Dependencies**

### **Core Framework**
- `next@^14.2.32` - React framework
- `react@^18` - React library
- `react-dom@^18` - React DOM

### **Styling**
- `tailwindcss@^3.3.0` - Utility-first CSS
- `autoprefixer@^10.0.1` - CSS autoprefixer
- `postcss@^8` - CSS processor

### **UI Components**
- `lucide-react@^0.294.0` - Icon library

### **Development**
- `typescript@^5` - TypeScript support
- `@types/node@^20` - Node.js types
- `@types/react@^18` - React types
- `@types/react-dom@^18` - React DOM types
- `eslint@^8` - Code linting
- `eslint-config-next@14.0.4` - Next.js ESLint config

## 🎨 **UI Components**

### **Main Interface**
- **Header**: Logo, title, and settings toggle
- **Settings Panel**: API key inputs and model selection
- **Upload Section**: File upload with progress indicators
- **Analysis Display**: Real-time results with formatted output

### **Analysis Output Formatting**
- **Header Banner**: Orange gradient with analysis title
- **Section Headers**: Bold, underlined styling
- **Bullet Points**: Orange bullet indicators
- **Timestamps**: Subtle time display
- **Error Messages**: Red error styling with detailed feedback

## 🔧 **Configuration**

### **Environment Variables**
```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### **API Configuration**
The frontend automatically detects the environment:
- **Development**: `http://localhost:8000`
- **Production**: Uses `NEXT_PUBLIC_API_URL`

## 📱 **Responsive Design**

### **Breakpoints**
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

### **Mobile Features**
- Touch-friendly upload interface
- Optimized form layouts
- Readable typography
- Accessible navigation

## 🎯 **File Upload Flow**

1. **File Selection**: User selects or drags log file
2. **Validation**: Frontend validates file type
3. **Upload**: File sent to backend API
4. **Processing**: Real-time progress indicators
5. **Analysis**: Streaming results display
6. **Results**: Formatted analysis output

## 🔄 **Real-time Features**

### **Streaming Responses**
- Real-time analysis progress
- Live result updates
- Progress indicators during processing
- Error handling with immediate feedback

### **State Management**
- File upload status tracking
- API key validation
- Message history management
- Loading states and error handling

## 🎨 **Styling System**

### **Design Tokens**
- **Primary Colors**: Orange to red gradients
- **Secondary Colors**: Blue accents
- **Neutral Colors**: Gray scale
- **Status Colors**: Green (success), Red (error), Orange (warning)

### **Component Styling**
- **Buttons**: Gradient backgrounds with hover effects
- **Cards**: Subtle shadows and rounded corners
- **Forms**: Clean inputs with focus states
- **Typography**: Clear hierarchy with proper spacing

## 🚀 **Deployment**

### **Vercel Deployment**
1. Connect GitHub repository to Vercel
2. Set environment variables:
   ```bash
   NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com
   ```
3. Deploy automatically on git push

### **Build Commands**
```bash
# Development
npm run dev

# Production build
npm run build
npm start

# Linting
npm run lint
```

## 🧪 **Testing**

### **Manual Testing**
- File upload with various log types
- API key validation
- Error handling scenarios
- Responsive design testing
- Real-time analysis flow

### **Browser Compatibility**
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## 🔒 **Security**

- API key handling with secure input
- File validation before upload
- CORS configuration for backend communication
- Input sanitization and validation

## 📈 **Performance**

- **Code Splitting**: Automatic route-based splitting
- **Image Optimization**: Next.js image optimization
- **Bundle Analysis**: Optimized bundle sizes
- **Caching**: Efficient caching strategies

## 🎯 **User Experience**

### **Accessibility**
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support
- Focus management

### **Error Handling**
- Clear error messages
- Graceful fallbacks
- User-friendly notifications
- Recovery suggestions

## 🔧 **Development**

### **Project Structure**
```
frontend/
├── app/
│   ├── globals.css      # Global styles
│   ├── layout.tsx       # Root layout
│   └── page.tsx         # Main page
├── package.json         # Dependencies
├── tailwind.config.js   # Tailwind configuration
└── tsconfig.json        # TypeScript configuration
```

### **Code Style**
- TypeScript for type safety
- ESLint for code quality
- Prettier for formatting
- Consistent component structure

---

**Built with Next.js, React, and Tailwind CSS for modern web experiences**