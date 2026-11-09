# tare.tools.agent-runtime

*English version: [README.md](README.md).*

O contexto delimitado canônico para a execução tática do loop de agente no ecossistema tare.tools.

Status: **núcleo tático, incremento isolado de sessão nativa, conversa imutável, admissão de prompt neutra em relação ao provedor e ingresso delimitado de grounding do SpecGraph**. O OS hospeda o runtime de agente atual por meio das portas de provedor e de processo do Kernel; este pacote não tem cache de prompt de provedor, persistência em sistema de arquivos, daemon, banco de dados, registro, UI nem sistema de plugins.

`tare.tools.agent-runtime` é o motor tático atual. Referências mais antigas ao
Aider descrevem integrações de host aposentadas. Este núcleo continua neutro em
relação ao provedor. Os limites de turnos e de tempo de relógio de uma tarefa
podem ser omitidos com `RunBudget()`; o cancelamento e as verificações estritas
de Work configuradas continuam ativos.

## Como ele se encaixa

[`tare.tools.os`](https://github.com/augusto-scarvalho/tare.tools.os) é o host.
Ele fixa este repositório como submódulo Git em `satellites/tare.tools.agent-runtime`
e fornece as portas de modelo e de ferramentas construídas sobre o
[`tare.tools.kernel`](https://github.com/augusto-scarvalho/tare.tools.kernel).
Este pacote não importa nenhum dos dois: tudo de que precisa é injetado.

## Instalar e testar

Você precisa de Python 3.11 ou mais recente e de Git. O pacote não tem
dependências de runtime; os testes rodam offline e usam fixtures sintéticas.

```bash
git clone https://github.com/augusto-scarvalho/tare.tools.agent-runtime.git
cd tare.tools.agent-runtime
python -m venv .venv
. .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src
```

A CI roda esses mesmos comandos em runners Linux e Windows hospedados pelo GitHub. Para
rodá-los nas suas próprias máquinas, defina as variáveis de repositório `CI_RUNNER_LINUX` e
`CI_RUNNER_WINDOWS` com uma lista de labels de runner (por exemplo `["self-hosted","Linux"]`).

## Escopo

Este repositório é dono de:

- progressão tática de turnos;
- contexto tático levado de um turno para o outro;
- admissão de prompt neutra em relação ao provedor, antes de invocar qualquer executor de turno;
- grounding imutável do repositório, injetado antes da execução no provedor;
- grounding imutável de Work, com cerca opcional e estrita contra drift e falta de progresso;
- eventos de conversa tática em memória, imutáveis byte a byte;
- snapshots atômicos da conversa, projeções determinísticas e um cache de
  projeções descartável;
- orçamentos de turnos, de novas tentativas e de tempo de relógio;
- estado de cancelamento cooperativo;
- construção e envio de uma reivindicação terminal com fence.

Ele não é dono de:

- Work, Workflow, Routing estratégico, Authority, Evidence nem liquidação (`tare.tools.os`);
- provedores, subprocessos, sandboxes nem execução mediada de capacidades (`tare.tools.kernel`);
- internals do Aider nem do histórico e congelado `tare.tools.harness`.

O núcleo consome um `TurnExecutor` e um `AuthorityGate` injetados, um
`PromptGate` opcional e um `GroundingEnvelope` opcional já materializado. Quando
nenhum prompt gate próprio é fornecido, o `PromptBudgetGate` delimitado fica
ativo. Cada turno recebe o binding opaco de Work e seu fence monotônico. O gate
de Authority é verificado de novo depois de uma invocação e antes do envio
terminal, para que um resultado atrasado não se torne canônico depois de um rebind.

## Incrementos de sessão nativa, contexto e handoff (0.11.0)

`NativeSession` conduz a progressão de modelo e ferramentas por meio de um
`ModelPort` e de um `CapabilityPort` injetados. Escolha-a para execução nativa;
não a aninhe dentro do loop agregado `AgentRuntime`. Os dois modos reutilizam os
contratos existentes de controle e de conversa. O OS fornece a composição de
Work e as tentativas duráveis; a integração de proposta até liquidação na
execução nativa continua pendente.

```python
from tare_tools_agent_runtime import NativeSession, NativeBudget, ConversationLog

session = NativeSession(
    session_id="session-1",
    instruction="Apply the scoped change and verify it.",
    capabilities=("read", "edit", "verify"),
    model=model_adapter,
    tools=capability_adapter,
    journal=ConversationLog(),
    budget=NativeBudget(max_model_calls=8, max_stagnant_turns=3),
)
outcome = session.run(execution_control)
```

Desde a 0.13.5, hosts podem fornecer `session.run(control, receive_message=receive)`.
O callback recebe os IDs de mensagens já registradas no journal e devolve o
próximo `(id, text)` ou `None`. Ele só roda entre rodadas completas ou depois de
uma proposta, com orçamento de execução disponível. O host guarda a entrada até
o journal conter o seu ID. A confirmação durável significa inserção no contexto,
não compreensão pelo modelo; invocações e lotes de ferramentas em andamento
terminam antes de a entrada ser consumida. Armazenamento, autenticação do
operador e UI continuam sendo responsabilidades do host.

`ModelReply` separa texto, pedidos completos de ferramenta e uma proposta
explícita de entrega. Uma resposta pode trazer até 32 pedidos de ferramenta,
executados um de cada vez na ordem fornecida
([lotes seriais de ferramentas](docs/NATIVE_BATCHES.md)).
`ToolCall.arguments_json` e os metadados e resultados do adaptador são objetos
JSON estritos e delimitados, materializados como strings imutáveis. O adaptador
deve normalizar os IDs de chamada de ferramenta para que sejam únicos na
sessão. O adaptador de capacidades verifica argumentos, versão esperada da
fonte e escopo, e recebe o mesmo controle vivo. Só o progresso observado pelo
adaptador zera a estagnação; texto em prosa não.

O journal registra a intenção antes de uma invocação e a observação depois
dela. Uma queda ou exceção entre as duas devolve `reconciliation_required` na
retomada; nem um pedido ao modelo nem um efeito incerto é repetido
automaticamente. Hosts que confirmam a limpeza local de uma invocação de modelo
podem chamar explicitamente
`abandon_interrupted_model(control, call_id=..., resolution_id=...)` antes de
retomar. Isso registra o abandono de uma resposta de modelo interrompida e
observada; a conclusão e o uso remotos continuam desconhecidos e a próxima
geração recebe um novo ID. A contagem original de pedidos é preservada, e
efeitos pendentes, resumos, interrupções não observadas e controle revogado,
cancelado ou expirado são recusados. IDs de resolução só são idempotentes para
a mesma chamada. Isso não recupera automaticamente um host que caiu nem desfaz
efeitos.

Um `ModelFailure(retryable=True)` explícito e concluído pode ser repetido dentro
do orçamento fixo. Observações de falha ficam no log; uso ausente não é zero.
Persistir uma resposta completa antes de uma queda permite consumir essa
resposta sem chamar o modelo de novo. Persistir o recibo de uma ferramenta
impede que ela seja repetida.

Para uma retomada simulada, construa uma nova sessão sobre o mesmo journal com
a mesma identidade, instrução, capacidades, orçamento e binding. Com
`NativeContext`, a política e as fontes protegidas também ficam vinculadas na
retomada. Chamadas usadas e a sequência atual de falhas continuam contadas (uma
retomada nunca as zera), incluindo as intenções auxiliares de resumo. `model_calls` conta intenções admitidas
(incluindo reservas interrompidas ou não despachadas), não execuções remotas
confirmadas nem chamadas faturáveis. Os metadados são guardados para
normalização futura e não formam um registro completo de custos.

Para um handoff explícito, o OS autentica a transferência causal com seu
registro de autoridade, mantém custódia exclusiva e chama `session.rebind(control,
previous_binding=session.binding_identity(), handoff_id=receipt_id)`. O novo
controle deve representar o sucessor atual. Isso acrescenta `binding_rebound`;
nunca reescreve o cabeçalho original nem concede autoridade por si só.
Invocações pendentes e sessões terminadas não podem transferir a execução.
Construa o adaptador de capacidades do sucessor com o novo binding e depois
rode o mesmo journal. Todas as chamadas anteriores de modelo e de resumo, IDs
de ferramenta duplicados e o contexto continuam contados. O wrapper do OS trata
a perda de uma confirmação de append lendo o binding atual antes de tentar a
transição de novo. Runtimes mais antigos não devem abrir um journal que contenha
a nova transição; journals anteriores ao handoff mantêm seu formato.

O [componente opcional de contexto nativo](docs/NATIVE_CONTEXT.md) admite o
pedido serializado completo contra limites explícitos de janela e de saída e
percentuais configuráveis. A compactação preserva os registros originais e só
ativa uma nova projeção depois de validá-la. Resumos que falham ou são incertos
não podem substituir em silêncio o contexto ativo. Sem esse componente, o
exemplo M1 acima não impõe a janela de contexto do provedor. A qualificação de
implantação real e de custo continua em aberto.

O host deve manter custódia exclusiva do journal entre instâncias e processos e
preservar seu prazo de execução na retomada. O lock local só impede reentrada
em uma instância. Persistência física, reconciliação de chamadas externas,
fencing entre processos, cancelamento no provedor e verificações independentes
de Work continuam sendo trabalho de integração. O adaptador precisa cooperar
com o cancelamento; este núcleo não consegue interromper uma chamada bloqueada
do sistema operacional. `proposed` é uma proposta de entrega, nunca um teste
aprovado, uma reivindicação terminal, um commit ou a admissão de um Work. Uma
proposta pode conter uma afirmação incorreta; o OS a avalia contra a evidência
original.

Este incremento não introduz dependência de Strands nem de Aider.

## Conversa imutável P1

`ConversationLog` materializa cada evento JSON aceito uma única vez, guarda seus
bytes UTF-8 canônicos exatos e o SHA-256, e publica o registro, a próxima
sequência e a revisão sob um único lock de escrita em memória e delimitado.
Snapshots e registros são imutáveis; as APIs de leitura devolvem valores
desacoplados.

`ProjectionSpec` seleciona um subconjunto ordenado dos eventos de origem. A
`ContextProjection` resultante é um ticket desacoplado, vinculado à revisão de
origem, à impressão digital do snapshot, à geração e à versão do projetor. A
impressão digital é metadado local de integridade, nunca Work nem Authority. Os
bytes da projeção são montados a partir dos bytes armazenados dos eventos,
guardados em cache apenas como otimização descartável, e reproduzíveis depois
que a estrutura do cache é substituída.

```python
from tare_tools_agent_runtime import ConversationLog, ProjectionSpec

conversation = ConversationLog()
conversation.append({"kind": "user", "text": "Inspect the failing test."})
conversation.append({"kind": "assistant", "text": "I found the race."})

ticket = conversation.project(
    ProjectionSpec(generation=1, projector_version="tactical-context-v1")
)
conversation.admit_current(ticket)
```

Isto não é cache de prompt do provedor. Serialização do pedido ao provedor,
posição dos pontos de cache, digests de prefixo, TTL, persistência e telemetria
de cache pertencem à fronteira de integração provedor/Kernel, fora deste núcleo.

## Admissão de prompt antes do provedor

`AgentRuntime.run()` materializa a instrução enviada uma única vez como um
`PromptEnvelope` imutável, com os bytes UTF-8 exatos e a identidade SHA-256. O
`PromptGate`, neutro em relação ao provedor, vê esse envelope antes do
`TurnExecutor`. Entradas negadas, malformadas ou com erro no gate devolvem
`OutcomeStatus.BLOCKED` com zero turnos e zero tentativas; nenhum adaptador de
provedor roda e nenhuma reivindicação terminal é enviada.

O digest permite verificações de identidade sem colocar texto puro nos
resultados ou nas representações; ele não é uma barreira de sigilo para prompts
de baixa entropia.

O `PromptBudgetGate` padrão admite prompts não vazios de até 16.384 bytes UTF-8.
O limite é configurável por injeção. Um gate devolve apenas `ALLOW` ou `DENY`;
de propósito, não existe campo para reescrever o prompt. Na admissão, o mesmo
objeto `PromptEnvelope` é anexado a cada nova tentativa e a cada pedido de turno.

```python
from tare_tools_agent_runtime import AgentRuntime, PromptBudgetGate

runtime = AgentRuntime(
    turn_executor=my_kernel_adapter,
    authority_gate=my_os_gate,
    prompt_gate=PromptBudgetGate(max_prompt_bytes=32_768),
)
```

Essa fronteira vale para todo provedor invocado pelo runtime gerenciado. Hooks
de prompt nativos de cada fornecedor só importam quando alguém usa a CLI de um
provedor diretamente, fora desse caminho de composição. Eventos internos de
ferramentas e de ciclo de vida do modelo continuam específicos de cada adaptador.

## Hooks canônicos de ferramentas no repositório

Este repositório é o segundo alvo opt-in de dogfood da política canônica de
hooks do Kernel. Seus manifests versionados para Claude Code, Codex e
Antigravity são adaptadores finos para o `tare-hook` instalado; não contêm
política local.

No `PreToolUse`, ferramentas nativas de grep e `rg`/`grep` no shell são
bloqueados em favor de `specgraph resolve <query> --format json`. O SpecGraph é
dono da recuperação delimitada em `MISS`/`STALE`/`TIMEOUT`/`UNAVAILABLE`, da
evidência externa descartável e da telemetria que preserva a privacidade. Antes
e depois de edições reconhecidas, a mesma política do Kernel fornece contexto
consultivo de impacto e de desatualização, onde o contrato do fornecedor permite.

O `specgraph.yaml` local mantém código e testes pesquisáveis e classifica os
documentos comuns do repositório apenas como `search_paths` neutros. De
propósito, ele não inventa specs formais, schemas, APIs nem ADRs. Instale
`tare-kernel` e `tare-tools-specgraph` no ambiente que inicia a CLI do
fornecedor; um comando de hook ausente não é substituído em silêncio por código
do repositório.

## Ingresso de grounding do SpecGraph

O host ou o adaptador do Kernel roda `specgraph ground` contra um manifest da
Library escolhido explicitamente e passa os bytes exatos do stdout para
`GroundingEnvelope`. O runtime valida JSON compacto canônico, status utilizável,
digest da fonte, contagens de entradas e o tamanho declarado em bytes. Depois
anexa o mesmo objeto imutável a cada `TurnRequest`; ele não mistura o grounding
na instrução nem na conversa tática, então a serialização para o provedor e a
posição do cache de prompt continuam sendo responsabilidade do Kernel.

```python
from tare_tools_agent_runtime import GroundingEnvelope

grounding = GroundingEnvelope(specgraph_completed_process.stdout)
outcome = runtime.run(
    binding=binding,
    instruction="Implement the accepted packet.",
    budget=budget,
    grounding=grounding,
)
```

Não há import de Library nem de SpecGraph, descoberta de manifest, acesso à rede
nem leitura de documentos de origem neste pacote. Resultados não utilizáveis do
SpecGraph são rejeitados antes de qualquer provedor executar.

## Controle estrito de trabalho gerenciado

`StrictWorkControl` é um modo opt-in que falha fechado. O host fornece um
`WorkEnvelope` exato de `backlog ground` e um `WorkGuard` injetado. O runtime
revalida esse escopo de trabalho antes do despacho, depois de cada resultado e
imediatamente antes de uma reivindicação terminal. Drift, revalidação
indisponível ou erro do guard interrompe a execução sem reivindicação canônica.

No modo estrito, todo resultado `CONTINUE` ou `SUCCEEDED` também precisa trazer
um `ProgressSignal` observado pelo Kernel: obrigações restantes, quantidade de
evidências verificadas e identidades de estado e de evidência. Regressão
bloqueia na hora. Janelas repetidas sem redução de obrigações e sem ganho de
evidência disparam `STAGNATION_DETECTED`; mudar o texto ou subdividir a tarefa
não é progresso. Um resultado `SUCCEEDED` fica bloqueado enquanto houver
obrigação em aberto. Adaptadores também podem devolver um turno `BLOCKED` com um
código de motivo estável; nesse caso o Runtime não emite reivindicação terminal,
inclusive fora do modo estrito de Work.

```python
runtime = AgentRuntime(
    turn_executor=my_kernel_adapter,
    authority_gate=my_os_gate,
    work_guard=my_backlog_guard,
)
outcome = runtime.run(
    binding=binding,
    instruction=instruction,
    budget=budget,
    work=work_envelope,
    strict_work=StrictWorkControl(max_stagnant_turns=2),
)
```

## Desenvolvimento

Instale e teste como mostrado em [Instalar e testar](#instalar-e-testar). O
pacote não tem dependências de runtime; o Strands fica de fora de propósito e
pode ser removido por construção. As regras para quem contribui estão em
[AGENTS.md](AGENTS.md).

## Exemplo mínimo

```python
from tare_tools_agent_runtime import AgentRuntime, BindingRef, RunBudget

runtime = AgentRuntime(turn_executor=my_kernel_adapter, authority_gate=my_os_gate)
outcome = runtime.run(
    binding=BindingRef(
        work_id="WORK-42",
        binding_id="binding-7",
        owner_id="worker-01",
        fence=7,
        token="opaque-authority-token",
    ),
    instruction="Implement the accepted packet.",
    budget=RunBudget(max_turns=4, max_retries_per_turn=1, wall_clock_seconds=300),
)
```

O host compõe essas dependências. O núcleo nunca as descobre nem as importa.

Os tetos da execução nativa são opções independentes. `NativeBudget(max_model_calls=None,
max_stagnant_turns=None, max_model_retries=None)` remove esses tetos
explicitamente; campos omitidos mantêm os padrões 16/3/0. `max_model_retries` conta
falhas seguidas: uma chamada de modelo que responde zera a contagem, então cada chamada
tem suas próprias novas tentativas.
`ContextPolicy(max_compactions=None, ...)` remove apenas o teto de número de
compactações. Chamadas, novas tentativas e resumos continuam observados; a
admissão de contexto e as verificações de cancelamento, binding e efeitos
pendentes continuam ativas. O host fornece e persiste qualquer prazo (ou nenhum
prazo). A continuação não tem um limite fixo separado de 64 mensagens; os
limites de tamanho de mensagem e de journal existentes continuam valendo.

Com `max_compactions=0`, os pedidos mantêm o histórico original e são admitidos
até `window - output_reserve - margin`. O gatilho suave apenas pede compactação;
ele não descarta contexto ainda utilizável quando a compactação está desligada.

### Observações do host (0.14.0)

Um host com custódia exclusiva do journal pode chamar
`continue_after_observation(control, observation_id=..., text=...,
after_sequence=..., after_sha256=...)` depois de uma proposta concluída. Isso
registra uma única observação do host, atribuída explicitamente e separada da
entrada autenticada do usuário. O predecessor precisa coincidir com o registro
terminal exato. O texto é limitado a 16 KiB. Uma repetição idêntica não faz nada;
observações conflitantes ou uma segunda observação falham. Controle original,
identidade do modelo, prazo e tetos cumulativos de chamadas permanecem. A
compactação preserva a observação. O OS decide a elegibilidade e a reconciliação
de fontes e efeitos; esta porta não autoriza leituras, não escolhe modelos e não
repete ferramentas. Adaptadores do Kernel pareados precisam renderizar
`host_observation` antes de um host habilitá-la.

### Pausa pelo host (0.14.1)

`NativeSession.run(control, pause_requested=host_pause_check)` aceita um callback
opcional sem argumentos. Um pedido verdadeiro devolve
`NativeOutcome(status="paused", reason="PAUSE_REQUESTED", model_calls=...)` só
depois de a resposta completa do modelo e seu lote original de ferramentas terem
sido consumidos, sem invocação inacabada nem resultado terminal pendente. Não faz
nova chamada ao modelo, não faz polling nem espera, e não recebe mensagens
humanas enfileiradas. Uma proposta terminal concluída continua `proposed`, mesmo
com pausa pedida, para que o host possa registrar o checkpoint da próxima etapa.
Cancelamento, revogação de binding, prazos e reconciliação de invocações
desconhecidas têm precedência. A pausa não grava evento no journal e não confere
autoridade.

Com custódia exclusiva do journal, reconstrua a sessão com o mesmo journal
validado e a mesma configuração e depois rode com a pausa falsa (ou omitida).
Ferramentas concluídas não são repetidas; chamadas cumulativas, orçamentos,
identidade e binding ficam iguais. O host precisa preservar o prazo original. A
política de checkpoint e retomada do workflow pertence ao host, não a esta porta
tática.

## Licença

O código do Agent Runtime é licenciado sob [Apache-2.0](LICENSE). Provedores,
ferramentas e outras dependências injetadas pelo host mantêm suas próprias
licenças.
