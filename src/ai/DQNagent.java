package ai;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class DQNagent {
    private int stateSize;
    private int actionSize;
    private double learningRate;
    private double gamma;
    private double epsilon;
    private double epsilonDecay;
    private double epsilonMin;
    private LinkedList<Experience> memory;
    private NeuralNetwork model;

    public DQNagent(int stateSize, int actionSize, double learningRate, double gamma, double epsilon,
                    double epsilonDecay, double epsilonMin) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.learningRate = learningRate;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.memory = new LinkedList<>();
        this.model = new NeuralNetwork(stateSize, actionSize);
    }

    public int act(int[] state) {
        if (Math.random() < epsilon) {
            return new Random().nextInt(actionSize); // Exploration
        } else {
            double[] qValues = model.predict(state);
            int maxIndex = 0;
            for (int i = 1; i < qValues.length; i++) {
                if (qValues[i] > qValues[maxIndex]) {
                    maxIndex = i;
                }
            }
            return maxIndex; // Exploitation
        }
    }

    public void remember(int[] state, int action, int reward, int[] nextState, boolean done) {
        memory.add(new Experience(state, action, reward, nextState, done));
    }

    public void replay(int batchSize) {
        if (memory.size() < batchSize) return;

        // Sample a batch of experiences
        List<Experience> batch = memory.subList(0, batchSize);
        
        for (Experience exp : batch) {
            int[] state = exp.state;
            int action = exp.action;
            int reward = exp.reward;
            int[] nextState = exp.nextState;
            boolean done = exp.done;

            double[] qValues = model.predict(state);
            double target = reward;
            if (!done) {
                double[] nextQValues = model.predict(nextState);
                target += gamma * Arrays.stream(nextQValues).max().getAsDouble(); // Q-value update
            }
            qValues[action] = target; // Update the Q-value for the action taken
            
            // Perform a gradient update (simple example, actual training would involve backpropagation)
            model.update(state, qValues);
        }

        // Decay epsilon
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }

    public static class Experience {
        int[] state;
        int action;
        int reward;
        int[] nextState;
        boolean done;

        public Experience(int[] state, int action, int reward, int[] nextState, boolean done) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }
    }
}
