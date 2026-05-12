"""
Gold standard test prompts for evaluation.

50 debate prompts across three difficulty tiers:
  - Easy: Thesis contains obvious logical fallacies
  - Medium: Legitimate argument with weak premises
  - Hard: Well-constructed argument requiring domain knowledge
"""

EASY_PROMPTS = [
    {
        "id": "easy_01",
        "text": "We should ban all cars because one person died in a car crash last week.",
        "expected_fallacies": ["faulty_generalization"],
        "topic": "policy",
    },
    {
        "id": "easy_02",
        "text": "Climate change can't be real because it was cold yesterday.",
        "expected_fallacies": ["faulty_generalization"],
        "topic": "science",
    },
    {
        "id": "easy_03",
        "text": "You can't criticize the military because you never served.",
        "expected_fallacies": ["ad_hominem"],
        "topic": "politics",
    },
    {
        "id": "easy_04",
        "text": "Everyone I know supports universal healthcare, so it must be the right policy.",
        "expected_fallacies": ["ad_populum"],
        "topic": "policy",
    },
    {
        "id": "easy_05",
        "text": "If we legalize marijuana, next people will want to legalize heroin and cocaine.",
        "expected_fallacies": ["fallacy_of_extension", "faulty_generalization", "false_causality"],
        "topic": "policy",
    },
    {
        "id": "easy_06",
        "text": "Professor Smith says homeopathy works, so it must be true.",
        "expected_fallacies": ["fallacy_of_credibility"],
        "topic": "science",
    },
    {
        "id": "easy_07",
        "text": "You're either with us or against us. There's no middle ground on immigration.",
        "expected_fallacies": ["false_dilemma"],
        "topic": "politics",
    },
    {
        "id": "easy_08",
        "text": "Capitalism is good because free markets are inherently beneficial.",
        "expected_fallacies": ["circular_reasoning"],
        "topic": "economics",
    },
    {
        "id": "easy_09",
        "text": "We shouldn't listen to climate scientists because they just want grant money.",
        "expected_fallacies": ["ad_hominem"],
        "topic": "science",
    },
    {
        "id": "easy_10",
        "text": "Think about the children who will suffer if we don't ban violent video games!",
        "expected_fallacies": ["appeal_to_emotion"],
        "topic": "media",
    },
    {
        "id": "easy_11",
        "text": "My grandfather smoked and lived to 95, so smoking can't be that bad for you.",
        "expected_fallacies": ["faulty_generalization"],
        "topic": "health",
    },
    {
        "id": "easy_12",
        "text": "If evolution is real, why are there still monkeys?",
        "expected_fallacies": ["fallacy_of_logic"],
        "topic": "science",
    },
    {
        "id": "easy_13",
        "text": "The government wants to control guns, which means they want to control everything about our lives.",
        "expected_fallacies": ["fallacy_of_extension", "faulty_generalization"],
        "topic": "politics",
    },
    {
        "id": "easy_14",
        "text": "Democracy must be the best system because most countries in the world use it.",
        "expected_fallacies": ["ad_populum"],
        "topic": "politics",
    },
    {
        "id": "easy_15",
        "text": "You support privacy rights? I bet you have something to hide.",
        "expected_fallacies": ["ad_hominem"],
        "topic": "rights",
    },
    {
        "id": "easy_16",
        "text": "Natural remedies are always better than pharmaceutical drugs because they come from nature.",
        "expected_fallacies": ["false_causality"],
        "topic": "health",
    },
    {
        "id": "easy_17",
        "text": "We've always done it this way, so there's no reason to change.",
        "expected_fallacies": ["fallacy_of_credibility"],
        "topic": "general",
    },
]

MEDIUM_PROMPTS = [
    {
        "id": "med_01",
        "text": (
            "Universal basic income reduces poverty because pilot programs in Finland "
            "and Kenya showed promising results with reduced stress and improved "
            "well-being among participants."
        ),
        "topic": "economics",
    },
    {
        "id": "med_02",
        "text": (
            "Social media should be regulated because studies show a correlation between "
            "heavy social media use and increased rates of depression among teenagers."
        ),
        "topic": "technology",
    },
    {
        "id": "med_03",
        "text": (
            "The death penalty should be abolished because innocent people have been "
            "wrongfully executed, and the system disproportionately affects minorities."
        ),
        "topic": "justice",
    },
    {
        "id": "med_04",
        "text": (
            "Artificial intelligence will create more jobs than it destroys, just as "
            "previous technological revolutions like the industrial revolution ultimately "
            "increased employment."
        ),
        "topic": "technology",
    },
    {
        "id": "med_05",
        "text": (
            "Immigration strengthens economies by filling labor shortages, increasing "
            "consumer spending, and bringing entrepreneurial talent. The US economy has "
            "historically benefited from waves of immigration."
        ),
        "topic": "economics",
    },
    {
        "id": "med_06",
        "text": (
            "Nuclear energy is the most practical solution to climate change because "
            "it produces zero carbon emissions during operation and has a higher energy "
            "density than any renewable source."
        ),
        "topic": "energy",
    },
    {
        "id": "med_07",
        "text": (
            "Free speech should have limits when it comes to hate speech because "
            "the harm caused to marginalized groups outweighs the speaker's right "
            "to express hateful views."
        ),
        "topic": "rights",
    },
    {
        "id": "med_08",
        "text": (
            "Mandatory voting would improve democracy by ensuring that elected officials "
            "represent the will of the entire population rather than just those motivated "
            "enough to vote."
        ),
        "topic": "politics",
    },
    {
        "id": "med_09",
        "text": (
            "Zoos are necessary for conservation because they fund breeding programs "
            "for endangered species and educate the public about biodiversity."
        ),
        "topic": "ethics",
    },
    {
        "id": "med_10",
        "text": (
            "College education should be free because it reduces inequality of opportunity "
            "and countries with free university education like Germany have strong economies."
        ),
        "topic": "education",
    },
    {
        "id": "med_11",
        "text": (
            "Veganism is the most ethical dietary choice because factory farming causes "
            "immense animal suffering and contributes significantly to greenhouse gas emissions."
        ),
        "topic": "ethics",
    },
    {
        "id": "med_12",
        "text": (
            "Restorative justice is more effective than punitive approaches because "
            "countries like Norway with low recidivism rates focus on rehabilitation "
            "rather than punishment."
        ),
        "topic": "justice",
    },
    {
        "id": "med_13",
        "text": (
            "Cryptocurrency will eventually replace traditional banking because "
            "decentralized finance eliminates the need for intermediaries and "
            "reduces transaction costs."
        ),
        "topic": "economics",
    },
    {
        "id": "med_14",
        "text": (
            "Remote work improves productivity because studies during the pandemic "
            "showed that most workers maintained or improved their output while "
            "working from home."
        ),
        "topic": "work",
    },
    {
        "id": "med_15",
        "text": (
            "Space exploration is worth the investment because the technologies "
            "developed for space missions frequently lead to practical applications "
            "that benefit everyday life."
        ),
        "topic": "science",
    },
    {
        "id": "med_16",
        "text": (
            "Genetic engineering of crops is safe and necessary to feed a growing "
            "world population. The scientific consensus supports the safety of GMOs."
        ),
        "topic": "science",
    },
]

HARD_PROMPTS = [
    {
        "id": "hard_01",
        "text": (
            "Kant's categorical imperative fails in trolley-problem scenarios because "
            "it cannot resolve duties in conflict. When saving five lives requires "
            "actively killing one, the universalizability test gives contradictory results "
            "depending on how the maxim is formulated."
        ),
        "topic": "ethics",
    },
    {
        "id": "hard_02",
        "text": (
            "Rawls's veil of ignorance is incoherent because rational agents behind "
            "the veil would need to know their risk preferences to choose principles "
            "of justice, but risk preferences are precisely the kind of particular "
            "knowledge the veil is supposed to exclude."
        ),
        "topic": "justice",
    },
    {
        "id": "hard_03",
        "text": (
            "The hard problem of consciousness shows that physicalism is false. No "
            "amount of functional or neurological explanation can account for why there "
            "is something it is like to experience qualia. The explanatory gap between "
            "physical processes and subjective experience is unbridgeable."
        ),
        "topic": "consciousness",
    },
    {
        "id": "hard_04",
        "text": (
            "Compatibilism is an evasion rather than a solution to the free will problem. "
            "By redefining 'free' to mean 'not externally coerced,' compatibilists "
            "sidestep the real question: whether our choices could have been otherwise "
            "given identical prior conditions."
        ),
        "topic": "free_will",
    },
    {
        "id": "hard_05",
        "text": (
            "Searle's Chinese Room argument decisively refutes strong AI. The person "
            "in the room manipulates symbols according to rules without understanding "
            "Chinese, proving that syntax is insufficient for semantics regardless "
            "of the system's complexity."
        ),
        "topic": "AI",
    },
    {
        "id": "hard_06",
        "text": (
            "Moral realism is untenable because the best explanation for moral "
            "disagreement across cultures is that moral facts don't exist. If objective "
            "moral truths were real, we would expect convergence over time, as we see "
            "in science, but moral progress looks more like shifting cultural norms "
            "than discovery of pre-existing truths."
        ),
        "topic": "metaethics",
    },
    {
        "id": "hard_07",
        "text": (
            "Kuhn's theory of paradigm shifts undermines scientific realism. If "
            "scientific theories are incommensurable across paradigms, then we cannot "
            "say science converges on truth — only that it replaces one internally "
            "coherent framework with another."
        ),
        "topic": "philosophy_of_science",
    },
    {
        "id": "hard_08",
        "text": (
            "Singer's utilitarian argument for animal rights fails because it relies "
            "on a sentience criterion that has no principled boundary. If we extend "
            "moral consideration to all sentient beings, we face an infinite regress "
            "about where sentience begins — insects? bacteria? — that makes the theory "
            "practically unworkable."
        ),
        "topic": "ethics",
    },
    {
        "id": "hard_09",
        "text": (
            "Bayesian epistemology cannot serve as a complete theory of rational belief "
            "because the choice of prior probabilities is inherently subjective. Two "
            "rational agents starting with different priors can reach radically different "
            "posterior beliefs from the same evidence, making rationality underdetermined."
        ),
        "topic": "epistemology",
    },
    {
        "id": "hard_10",
        "text": (
            "Game theory's Nash equilibrium fails as a predictive model of human "
            "behavior because humans consistently cooperate in prisoner's dilemma "
            "scenarios where Nash equilibrium predicts defection. This isn't irrationality "
            "— it's evidence that rationality itself is more social than game theory allows."
        ),
        "topic": "decision_theory",
    },
    {
        "id": "hard_11",
        "text": (
            "Nozick's libertarianism is self-defeating because the minimal state he "
            "advocates cannot maintain the property rights it exists to protect without "
            "taxation, and taxation violates the very self-ownership principles the "
            "theory is built on."
        ),
        "topic": "political_philosophy",
    },
    {
        "id": "hard_12",
        "text": (
            "The reliabilist theory of knowledge fails because it cannot solve the "
            "generality problem — any belief-forming process can be described at multiple "
            "levels of generality, each with different reliability ratings, and the theory "
            "provides no principled way to select the right level."
        ),
        "topic": "epistemology",
    },
    {
        "id": "hard_13",
        "text": (
            "Mill's harm principle is insufficient as a basis for free speech because "
            "the concept of 'harm' is inherently contested. If psychological harm counts, "
            "then virtually any speech can be restricted; if only physical harm counts, "
            "then incitement to violence through gradual radicalization goes unaddressed."
        ),
        "topic": "political_philosophy",
    },
    {
        "id": "hard_14",
        "text": (
            "Arrow's impossibility theorem proves that democratic decision-making is "
            "fundamentally flawed. No voting system can simultaneously satisfy all "
            "fairness criteria, which means every democratic process is vulnerable to "
            "manipulation or produces paradoxical outcomes."
        ),
        "topic": "social_choice",
    },
    {
        "id": "hard_15",
        "text": (
            "The personal identity problem shows that punishment is fundamentally "
            "unjust. If psychological continuity is what makes someone the same person, "
            "then a criminal who has undergone radical psychological change is literally "
            "a different person from the one who committed the crime."
        ),
        "topic": "identity",
    },
    {
        "id": "hard_16",
        "text": (
            "Consequentialism collapses under the problem of demandingness. If we are "
            "morally required to maximize good outcomes, then any expenditure on our own "
            "pleasure while others suffer is immoral. This makes the theory either too "
            "demanding to follow or requires ad hoc modifications that undermine its "
            "theoretical elegance."
        ),
        "topic": "ethics",
    },
    {
        "id": "hard_17",
        "text": (
            "Popper's falsificationism fails as a demarcation criterion because "
            "Duhem-Quine underdetermination means no individual hypothesis can be "
            "conclusively falsified — any apparent falsification can be deflected by "
            "modifying auxiliary hypotheses. This makes the boundary between science "
            "and non-science far blurrier than Popper claimed."
        ),
        "topic": "philosophy_of_science",
    },
]

ALL_PROMPTS = EASY_PROMPTS + MEDIUM_PROMPTS + HARD_PROMPTS

# Convenience accessors
def get_prompts_by_tier(tier):
    return {"easy": EASY_PROMPTS, "medium": MEDIUM_PROMPTS, "hard": HARD_PROMPTS}[tier]

def get_prompt_by_id(prompt_id):
    for p in ALL_PROMPTS:
        if p["id"] == prompt_id:
            return p
    return None
