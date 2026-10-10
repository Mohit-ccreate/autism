/**
 * Service Worker for NEURO-SCREEN AI Offline PWA
 * Caches all local assets so the application functions 100% offline.
 */

const CACHE_NAME = 'neuroscreen-v1.0';
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './manifest.json',
  './metrics.json',
  './models_export.json',
  './vendor/three.min.js',
  './vendor/chart.umd.min.js',
  './vendor/gsap.min.js',
  './vendor/jspdf.umd.min.js',
  './vendor/ort.min.js'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request).catch(() => {
        if (event.request.destination === 'document') {
          return caches.match('./index.html');
        }
      });
    })
  );
});
