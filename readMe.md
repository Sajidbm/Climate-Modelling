# Climate Modeling: Error Propagation and Regime Change Detection

This is a project focused on understanding error propagation in atmospheric flow simulations and developing regime change detection schemes for Earth systems models.

## Project Overview

This project investigates how errors propagate through simple simulation models that replicate multi-state atmospheric flows. The ultimate goal is to construct robust regime change detection schemes applicable to Earth systems models.

## Objectives

1. **Error Propagation Analysis**: Study how computational and modeling errors propagate through dynamical systems that exhibit multi-state atmospheric behavior.
2. **Regime Change Detection**: Develop and validate detection schemes that can identify transitions between different atmospheric states.
3. **Neural Emulator Development**: Build and evaluate neural network-based emulators for atmospheric models

## Current Focus

I am currently exploring neural emulators for toy atmospheric models, starting with the **Lorenz '63 system**. This foundational work helps establish methodologies before scaling to more complex systems.

### Study Directory
- `lorenz63.py` - Basic Lorenz 63 implementation
- `lorenz63JAX.py` - JAX-accelerated version for parallel computation
- `parallel_mapping.py` - Utilities for parallel ensemble simulations

## Planned Progression

1. **Phase 1** (Current): Lorenz 63 system - Understanding chaotic dynamics and neural emulation.
2. **Phase 2**: Lorenz 96 model - Multi-scale atmospheric processes.
3. **Phase 3**: Shallow water equations - More realistic atmospheric flow dynamics.

## Motivation

I am inspired by the work of Google's NeuralGCM. My research aims to advance our understanding of how machine learning can improve climate model accuracy and computational efficiency. The Lorenz systems serve as standard benchmarks in the climate modeling community for testing algorithms that parametrize unresolved processes in global climate models.