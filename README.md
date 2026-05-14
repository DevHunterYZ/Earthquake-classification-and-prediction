# Earthquake-classification-and-prediction
Earthquake classification and prediction with GNN

This system watches for big earthquakes, then guesses how many smaller quakes (aftershocks) will follow.

When you pick a mainshock — either from the live USGS feed or by typing one in — the system runs three simple math models. Omori-Utsu says aftershocks are like a crowd leaving a stadium: most rush out right away, then the flow slows down over days. Gutenberg-Richter says for every big quake, you get about 10 times as many quakes one magnitude smaller. Båth's Law says the biggest aftershock is usually about 1.2 magnitude units below the mainshock. The system combines these to tell you things like "expect ~170 quakes between M2-M3 in the next 30 days, with the busiest window being the first 24 hours (~35 quakes), and the largest aftershock will likely be around M5.3." It also gives a risk score from 0-100 based on how many M5+ aftershocks are expected.
