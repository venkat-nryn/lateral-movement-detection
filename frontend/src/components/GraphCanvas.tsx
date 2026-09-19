"use client";

import { useEffect, useRef, useState } from "react";
import type { Core } from "cytoscape";
import type { GraphEdge, GraphNode } from "@/lib/api";

type StyleRule = { selector: string; style: Record<string, unknown> };

const STYLE: StyleRule[] = [
  {
    selector: "node",
    style: {
      "background-color": "#16324a",
      "border-width": 1.2,
      "border-color": "#38bdf8",
      width: "data(size)",
      height: "data(size)",
      label: "",
      "font-size": 8,
      "font-family": "ui-monospace, Consolas, monospace",
      color: "#d9dfe8",
      "text-valign": "bottom",
      "text-margin-y": 3,
      "text-outline-width": 2,
      "text-outline-color": "#0a0d12",
      "transition-property": "opacity, border-color, border-width",
      "transition-duration": 180,
    },
  },
  { selector: 'node[kind = "user"]', style: { shape: "diamond", "background-color": "#2a2150", "border-color": "#a78bfa" } },
  { selector: "node[alerts > 0]", style: { "border-color": "#f43f5e", "border-width": 2.6, label: "data(label)" } },
  { selector: "node.hover, node:selected, node.focus", style: { label: "data(label)", "border-width": 3, "border-color": "#f8fafc", "z-index": 20 } },
  {
    selector: "edge",
    style: {
      width: "data(width)",
      "line-color": "#2e3d50",
      "target-arrow-color": "#2e3d50",
      "target-arrow-shape": "triangle",
      "arrow-scale": 0.7,
      "curve-style": "bezier",
      opacity: 0.85,
      "transition-property": "opacity, line-color",
      "transition-duration": 180,
    },
  },
  { selector: 'edge[relation = "identity"]', style: { "line-style": "dashed", "line-color": "#43386e", "target-arrow-color": "#43386e", opacity: 0.55 } },
  { selector: "edge[alerts > 0]", style: { "line-color": "#f43f5e", "target-arrow-color": "#f43f5e", opacity: 1, "z-index": 10, width: 3 } },
  { selector: "edge[ground_truth > 0].truth", style: { "line-color": "#a78bfa", "target-arrow-color": "#a78bfa", width: 3.2, opacity: 1, "z-index": 11 } },
  { selector: "edge:selected", style: { "line-color": "#f8fafc", "target-arrow-color": "#f8fafc", width: 3.5 } },
  { selector: ".faded", style: { opacity: 0.07 } },
];

export interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  showIdentity: boolean;
  highlightSuspicious: boolean;
  showTruth?: boolean;
  selectedId: string | null;
  fitKey: string;
  onSelectNode: (id: string) => void;
  onSelectEdge: (id: string) => void;
  onClear: () => void;
}

export function GraphCanvas(props: GraphCanvasProps) {
  const { nodes, edges, showIdentity, highlightSuspicious, showTruth = false, selectedId, fitKey } = props;
  const container = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const handlers = useRef(props);
  handlers.current = props;
  const lastFit = useRef<string>("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let disposed = false;
    void (async () => {
      const cytoscape = (await import("cytoscape")).default;
      if (disposed || !container.current) return;
      const instance = cytoscape({
        container: container.current,
        elements: [],
        style: STYLE as never,
        wheelSensitivity: 0.25,
        minZoom: 0.1,
        maxZoom: 5,
        boxSelectionEnabled: false,
      });
      instance.on("tap", "node", (event) => handlers.current.onSelectNode(event.target.id()));
      instance.on("tap", "edge", (event) => handlers.current.onSelectEdge(event.target.id()));
      instance.on("tap", (event) => {
        if (event.target === instance) handlers.current.onClear();
      });
      instance.on("mouseover", "node", (event) => event.target.addClass("hover"));
      instance.on("mouseout", "node", (event) => event.target.removeClass("hover"));
      cy.current = instance;
      setReady(true);
    })();
    return () => {
      disposed = true;
      cy.current?.destroy();
      cy.current = null;
    };
  }, []);

  // Sync elements, keeping existing positions so refreshes do not jump.
  useEffect(() => {
    const instance = cy.current;
    if (!instance) return;
    const keptNodes = showIdentity ? nodes : nodes.filter((n) => n.kind === "host");
    const keptIds = new Set(keptNodes.map((n) => n.id));
    const keptEdges = edges.filter(
      (e) => (showIdentity || e.relation === "movement") && keptIds.has(e.source) && keptIds.has(e.target),
    );
    const maxEvents = Math.max(1, ...keptNodes.map((n) => n.events ?? 1));
    const wanted = new Set<string>([...keptIds, ...keptEdges.map((e) => e.id)]);
    const added: string[] = [];

    instance.batch(() => {
      instance.elements().forEach((element) => {
        if (!wanted.has(element.id())) element.remove();
      });
      for (const node of keptNodes) {
        const data = {
          id: node.id,
          label: node.label,
          kind: node.kind,
          events: node.events ?? 1,
          alerts: node.alerts ?? 0,
          size: 11 + 30 * Math.sqrt((node.events ?? 1) / maxEvents),
        };
        const existing = instance.getElementById(node.id);
        if (existing.nonempty()) existing.data(data);
        else {
          instance.add({ group: "nodes", data });
          added.push(node.id);
        }
      }
      for (const edge of keptEdges) {
        const data = { ...edge, width: 0.8 + Math.min(5, Math.log2(1 + edge.events)) };
        const existing = instance.getElementById(edge.id);
        if (existing.nonempty()) existing.data(data);
        else instance.add({ group: "edges", data });
      }
    });

    // Seed each new node beside an already-placed neighbour.
    for (const id of added) {
      const node = instance.getElementById(id);
      const anchor = node.neighborhood("node").filter((n) => !added.includes(n.id())).first();
      if (anchor.nonempty()) {
        const box = anchor.boundingBox({});
        const x = (box.x1 + box.x2) / 2;
        const y = (box.y1 + box.y2) / 2;
        node.position({ x: x + (Math.random() - 0.5) * 60, y: y + (Math.random() - 0.5) * 60 });
      }
    }

    const refit = lastFit.current !== fitKey;
    if (added.length || refit) {
      instance
        .layout({
          name: "cose",
          animate: !refit,
          animationDuration: 400,
          randomize: refit,
          fit: refit,
          padding: 30,
          nodeRepulsion: () => 12000,
          idealEdgeLength: () => 70,
          numIter: 700,
        } as never)
        .run();
      lastFit.current = fitKey;
    }
  }, [nodes, edges, showIdentity, fitKey, ready]);

  // Suspicious-path highlight, ground-truth overlay and selection.
  useEffect(() => {
    const instance = cy.current;
    if (!instance) return;
    instance.batch(() => {
      instance.elements().removeClass("faded focus truth");
      if (showTruth) instance.edges("[ground_truth > 0]").addClass("truth");
      if (highlightSuspicious) {
        const suspicious = instance.edges("[alerts > 0]");
        instance.elements().addClass("faded");
        suspicious.removeClass("faded");
        suspicious.connectedNodes().removeClass("faded");
      }
      instance.elements(":selected").unselect();
      if (selectedId) {
        const element = instance.getElementById(selectedId);
        if (element.nonempty()) {
          element.select();
          if (element.isNode()) element.neighborhood().removeClass("faded").addClass("focus");
        }
      }
    });
  }, [highlightSuspicious, showTruth, selectedId, nodes, edges, showIdentity, ready]);

  return (
    <div className="relative h-full w-full">
      <div ref={container} className="absolute inset-0" />
      <div className="pointer-events-none absolute bottom-2 left-2 flex gap-2">
        {[
          ["+", () => cy.current?.zoom({ level: (cy.current?.zoom() ?? 1) * 1.25, renderedPosition: center(cy.current) })],
          ["−", () => cy.current?.zoom({ level: (cy.current?.zoom() ?? 1) / 1.25, renderedPosition: center(cy.current) })],
          ["Fit", () => cy.current?.fit(undefined, 30)],
        ].map(([label, action]) => (
          <button
            key={label as string}
            onClick={action as () => void}
            className="pointer-events-auto h-7 min-w-7 rounded border border-soc-line bg-soc-panel/90 px-2 text-[12px] text-soc-muted backdrop-blur transition-colors hover:text-soc-text"
          >
            {label as string}
          </button>
        ))}
      </div>
    </div>
  );
}

function center(instance: Core | null) {
  if (!instance) return { x: 0, y: 0 };
  return { x: instance.width() / 2, y: instance.height() / 2 };
}
