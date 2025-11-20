# 2. EXISTING SYSTEMS

## 2.1 Literature Survey

The existing health and fitness chatbot systems primarily include:

### **• Nutrition Tracking Apps (MyFitnessPal, Cronometer)**
These apps provide calorie counting and meal logging but lack interactivity. They require manual data entry for every meal and exercise, making them tedious to use. They focus on tracking rather than guidance and do not offer conversational interfaces or real-time personalized advice.

### **• Health Assessment Chatbots (Ada Health, Symptom Checker Apps)**
These focus on symptom analysis and medical diagnosis rather than lifestyle guidance. They require extensive user input through structured questionnaires and lack natural conversational flow. They are not designed for fitness planning or nutritional recommendations.

### **• AI Health Coaches (Lark Health, Noom)**
While these provide some AI-driven guidance, they are primarily text-based and require continuous data tracking. They lack voice input/output capabilities and often use subscription-based models with complex setup processes. Many rely on predefined coaching scripts rather than dynamic AI responses.

### **• Generic Voice Assistants (Google Assistant, Alexa Skills)**
Generic voice assistants offer basic health features like activity logging and workout timers, but they are not specialized for comprehensive health guidance. They provide fragmented skill ecosystems with no unified platform for nutrition, fitness, and wellness. Natural language understanding is limited to simple commands.

### **• Chatbots with Fixed Responses**
Many existing health chatbots use rule-based systems with predefined responses. They fail to handle diverse, open-ended queries and cannot adapt to user context or conversation history. They lack the ability to generate personalized meal plans or workout routines dynamically.

---

## Key Limitations of Existing Systems

These systems often rely on **pre-stored data** and **rule-based logic**, making them unable to understand open-ended or context-specific health queries.

They lack:
- **Natural Language Understanding** – Cannot comprehend conversational, nuanced questions
- **Voice Interaction** – Most are text-only with no voice input/output support
- **Personalized Engagement** – Generic advice without considering individual goals, preferences, or history
- **Adaptability** – Cannot learn from user interactions or adjust recommendations dynamically
- **Comprehensive Guidance** – Focus on either fitness OR nutrition, not both holistically
- **Accessibility** – Poor support for users with disabilities who need voice interfaces

Existing AI health assistants (like Google Assistant or Alexa) are **generic and not specialized** for comprehensive health and wellness guidance — making them unsuitable for personalized nutrition, fitness planning, and lifestyle recommendations.

---

## Proposed System Advantages

The proposed **Healthy Habits Chatbot** aims to overcome these limitations using a **hybrid AI approach** that combines:

1. **Machine Learning Classification (TF-IDF)** – For fast, efficient intent recognition of common health queries
2. **Generative AI (Gemini LLM)** – For intelligent, context-aware responses to complex, open-ended questions
3. **Web Speech API** – For complete voice input/output, enabling hands-free interaction and accessibility
4. **Personalized Health Planning** – Dynamic BMI calculation, calorie recommendations, and goal-specific meal/workout plans
5. **Conversational Interface** – Natural, chat-based interaction with real-time response generation

This hybrid approach ensures the system can handle both structured queries (through ML classification) and unstructured, conversational questions (through Gemini LLM), providing a superior user experience with complete voice integration and holistic health guidance.

---

### References

1. Fitzpatrick, K.K., Darcy, A., & Vierhile, M. (2017). "Delivering Cognitive Behavior Therapy to Young Adults With Symptoms of Depression and Anxiety Using a Fully Automated Conversational Agent (Woebot)." *JMIR Mental Health*, 4(2), e19.

2. Zeevi, D., et al. (2015). "Personalized Nutrition by Prediction of Glycemic Responses." *Cell*, 163(5), 1079-1094.

3. Google DeepMind (2024). "Gemini: A Family of Highly Capable Multimodal Models." *Technical Report*.

4. W3C Speech API Community Group (2023). "Web Speech API Specification." *World Wide Web Consortium*.

5. Laranjo, L., et al. (2018). "Conversational agents in healthcare: a systematic review." *Journal of the American Medical Informatics Association*, 25(9), 1248-1258.

6. Greer, S., et al. (2019). "Use of the chatbot 'Vivibot' to deliver positive psychology skills and promote well-being among young people after cancer treatment." *JMIR mHealth and uHealth*, 7(10), e15018.

7. Ni, L., et al. (2017). "The effect of a mobile health application on eating behaviors and weight loss." *Journal of Medical Internet Research*, 19(4), e92.

8. Tudor Car, L., et al. (2020). "Conversational Agents in Health Care: Scoping Review and Conceptual Analysis." *Journal of Medical Internet Research*, 22(8), e17158.
