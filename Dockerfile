# Gunakan base image Node.js
FROM node:18.16

# Set direktori kerja
WORKDIR /usr/src/app

# Salin semua file
COPY . .

# Install dependencies
RUN npm install

# Expose port
EXPOSE 8080

# Jalankan aplikasi
CMD ["node", "index.js"]
